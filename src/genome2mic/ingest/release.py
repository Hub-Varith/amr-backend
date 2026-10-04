"""Import a frozen team data release into a run root: ``import-release``.

A release is a directory of already-built stage outputs (labels, known-AMR features,
lineages, frozen splits, ...) plus a ``SHA256SUMS`` file. Importing it lets
``train``, ``evaluate`` and ``report`` run on real data without re-running ingest,
QC, AMRFinderPlus, lineages or splits here. Only the *data* is ported; no model or
code from the releasing branch is used.

Sources (exactly one):

* ``release_dir``: a local copy (for example the develop checkout's
  ``data/processed/``); it is only ever read.
* ``s3_release``: a release name; ``aws s3 sync s3://g2m-data-v1/releases/<NAME>/
  <root>/data/release_<NAME>/`` runs first (``aws_profile`` sets ``AWS_PROFILE`` for
  that subprocess only), then the synced directory is imported like a local one.

Steps:

1. Verify every file listed in ``SHA256SUMS`` (sha256). Any mismatch, any listed file
   missing, or a required file not covered by ``SHA256SUMS`` stops the import
   (:class:`ReleaseError`); nothing is written.
2. Refuse when the run root already holds a ``splits.parquet`` that differs from the
   release's (CLAUDE.md rule 7: splits are frozen and are never replaced).
3. Refuse when a drug of the release's ``pairs_kept.csv`` is not a canonical drug
   name in ``configs/drugs.yaml``.
4. Copy the release files byte-for-byte into ``<root>/data/processed/`` (content is
   never modified) and re-verify the copies.
5. Write ``qc.parquet`` (one row per genome of ``known_amr.parquet``: ``qc_pass`` True,
   QC numbers and ``qc_fail_reason`` null; the release ships no assemblies, so QC was
   not assessed here) and ``IMPORTED_RELEASE.json`` (release, source, verification,
   time, notes).

What a release does not provide, and what the pipeline does instead: no Mash sketches
(``nearest_training_distance`` is null), no unitig matrix (unitig models are skipped),
no per-genome tool output (``b0_resfinder`` is skipped), no species reference genomes
(the model bundle has no ``reference_sketches.npz``, so it cannot identify the species
of a new FASTA until references are added).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from genome2mic.droplog import DropLog
from genome2mic.errors import ContractViolation, Genome2MicError, ToolNotAvailable
from genome2mic.io import write_parquet
from genome2mic.paths import Paths

logger = logging.getLogger(__name__)

STAGE = "import-release"
S3_RELEASES_PREFIX = "s3://g2m-data-v1/releases"
SHA256SUMS = "SHA256SUMS"
IMPORTED_RELEASE_FILE = "IMPORTED_RELEASE.json"

RELEASE_FILES: tuple[str, ...] = (
    "labels.parquet",
    "known_amr.parquet",
    "known_amr_columns.csv",
    "lineages.parquet",
    "splits.parquet",
    "pairs_kept.csv",
    "label_counts.csv",
    "RELEASE",
    SHA256SUMS,
    "download_manifest.json",
    "tool_versions.json",
)
"""Files copied byte-for-byte into ``<root>/data/processed/`` when present."""

REQUIRED_FILES: tuple[str, ...] = (
    "labels.parquet",
    "known_amr.parquet",
    "lineages.parquet",
    "splits.parquet",
    "pairs_kept.csv",
)
"""Files the pipeline cannot run without; each must also be covered by ``SHA256SUMS``."""

NOTES: tuple[str, ...] = (
    "QC not assessed: release ships no assemblies",
    "no Mash sketches: nearest_training_distance unavailable",
    "no unitig matrix",
)

AwsRunner = Callable[..., Any]


class ReleaseError(Genome2MicError):
    """The release cannot be imported (checksum mismatch, missing file, unknown drug, ...)."""


@dataclass
class ImportSummary:
    """What :func:`run` did."""

    release: str | None
    source_dir: Path
    files: list[str]
    n_genomes_qc: int
    n_pairs: int
    sha256_verified: bool = True
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Checksums
# --------------------------------------------------------------------------- #


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, streamed in 1 MiB chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sha256sums(path: Path) -> dict[str, str]:
    """``file name -> hex digest`` from a ``sha256sum``-style file (``<hex>  <name>``; ``*name`` allowed)."""
    out: dict[str, str] = {}
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64 or any(c not in "0123456789abcdefABCDEF" for c in parts[0]):
            raise ReleaseError(f"{path}: line {line_no} is not '<sha256>  <file>': {raw!r}")
        name = parts[1].strip().lstrip("*")
        if "/" in name or name in ("", ".", ".."):
            raise ReleaseError(f"{path}: line {line_no}: unexpected file name {name!r}")
        if name in out:
            raise ReleaseError(f"{path}: {name} listed twice")
        out[name] = parts[0].lower()
    if not out:
        raise ReleaseError(f"{path}: no checksums")
    return out


def verify_release(release_dir: Path) -> dict[str, str]:
    """Check every ``SHA256SUMS`` entry of ``release_dir``; return the verified ``name -> digest``.

    Raises:
        ReleaseError: ``SHA256SUMS`` missing, a listed file missing or with a different
            digest, or a :data:`REQUIRED_FILES` entry not listed (it could not be verified).
    """
    release_dir = Path(release_dir)
    sums_path = release_dir / SHA256SUMS
    if not sums_path.is_file():
        raise ReleaseError(f"{sums_path} not found: a release must ship SHA256SUMS")
    expected = parse_sha256sums(sums_path)
    unlisted = [name for name in REQUIRED_FILES if name not in expected]
    if unlisted:
        raise ReleaseError(f"{sums_path}: required file(s) {unlisted} are not covered by SHA256SUMS")
    problems: list[str] = []
    for name, digest in sorted(expected.items()):
        path = release_dir / name
        if not path.is_file():
            problems.append(f"{name}: listed in SHA256SUMS but missing")
            continue
        actual = sha256_file(path)
        if actual != digest:
            problems.append(f"{name}: sha256 {actual[:12]}... != SHA256SUMS {digest[:12]}...")
    if problems:
        raise ReleaseError(f"release {release_dir} failed verification; nothing imported: " + "; ".join(problems))
    logger.info("[%s] %s: %d file(s) match SHA256SUMS", STAGE, release_dir, len(expected))
    return expected


# --------------------------------------------------------------------------- #
# S3 download
# --------------------------------------------------------------------------- #


def s3_release_uri(name: str) -> str:
    return f"{S3_RELEASES_PREFIX}/{name}/"


def _check_release_name(name: str) -> str:
    text = str(name).strip()
    if not text or "/" in text or text in (".", "..") or any(c.isspace() for c in text):
        raise ValueError(f"invalid release name {name!r}")
    return text


def download_s3_release(
    paths: Paths,
    name: str,
    *,
    aws_profile: str | None = None,
    runner: AwsRunner | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> Path:
    """``aws s3 sync s3://g2m-data-v1/releases/<name>/ <root>/data/release_<name>/``; returns the directory.

    ``runner`` defaults to :func:`subprocess.run` (tests pass a fake; nothing here
    touches S3 by itself). ``aws_profile`` is set as ``AWS_PROFILE`` for the
    subprocess only.
    """
    from genome2mic.ingest.fetch import subprocess_env, validate_aws_profile  # noqa: PLC0415

    name = _check_release_name(name)
    if aws_profile is not None:
        validate_aws_profile(aws_profile)
    if which("aws") is None:
        raise ToolNotAvailable("aws", "Install the AWS CLI v2 and log in (aws sso login / aws configure) to download a release.")
    dest = paths.data_dir / f"release_{name}"
    dest.mkdir(parents=True, exist_ok=True)
    command = ["aws", "s3", "sync", s3_release_uri(name), str(dest)]
    logger.info("[%s] %s%s", STAGE, " ".join(command), f" (AWS_PROFILE={aws_profile})" if aws_profile else "")
    run_command = runner if runner is not None else subprocess.run
    result = run_command(command, check=False, env=subprocess_env(aws_profile))
    code = getattr(result, "returncode", 0)
    if code:
        raise ReleaseError(f"aws s3 sync of {s3_release_uri(name)} failed with exit code {code}")
    return dest


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #


def _release_name(release_dir: Path) -> str | None:
    path = release_dir / "RELEASE"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "release" and value.strip():
            return value.strip()
    return None


def _check_frozen_splits(paths: Paths, release_dir: Path) -> None:
    """Refuse to replace an existing, different ``splits.parquet`` (rule 7)."""
    target = paths.splits
    if target.is_file() and sha256_file(target) != sha256_file(release_dir / "splits.parquet"):
        raise ContractViolation(
            f"{target} exists and differs from the release's splits.parquet; splits are frozen "
            "(CLAUDE.md rule 7). Import into a fresh --root instead.",
            STAGE,
        )


def _check_drugs(release_dir: Path, config: Any) -> int:
    """Every drug of the release's ``pairs_kept.csv`` must be a canonical ``drugs.yaml`` name."""
    pairs = pd.read_csv(release_dir / "pairs_kept.csv", dtype=str, keep_default_na=False)
    for column in ("species", "drug"):
        if column not in pairs.columns:
            raise ReleaseError(f"pairs_kept.csv has no {column!r} column")
    if config is not None:
        unknown = sorted({d for d in pairs["drug"] if config.normalize_drug(d) != d})
        if unknown:
            raise ReleaseError(
                f"release pairs_kept.csv drug(s) {unknown} are not canonical names in drugs.yaml; "
                "add them (develop's names are the team standard) before importing"
            )
        species = sorted({s for s in pairs["species"] if s not in config.species})
        if species:
            raise ReleaseError(f"release pairs_kept.csv species {species} are not in species.yaml")
    return len(pairs)


def qc_table(known_amr: pd.DataFrame) -> pd.DataFrame:
    """``qc.parquet`` for a release: one passing row per known-AMR genome, QC numbers null."""
    from genome2mic.qc import QC_COLUMNS  # noqa: PLC0415

    ids = known_amr["genome_id"].astype(str)
    if ids.duplicated().any():
        raise ContractViolation(f"known_amr.parquet has {int(ids.duplicated().sum())} duplicate genome_id(s)", STAGE)
    n = len(ids)
    frame = pd.DataFrame(
        {
            "genome_id": pd.array(ids.tolist(), dtype="str"),
            "species": pd.array(known_amr["species"].astype(str).tolist(), dtype="str"),
            "n_contigs": pd.array([None] * n, dtype="Int64"),
            "total_length": pd.array([None] * n, dtype="Int64"),
            "n50": pd.array([None] * n, dtype="Int64"),
            "gc_percent": pd.array([None] * n, dtype="Float64"),
            "mash_species": pd.array([None] * n, dtype="str"),
            "mash_distance": pd.array([None] * n, dtype="Float64"),
            "qc_pass": [True] * n,
            "qc_fail_reason": pd.array([None] * n, dtype="str"),
        }
    )
    frame["qc_pass"] = frame["qc_pass"].astype(bool)
    return frame[list(QC_COLUMNS)]


def run(
    paths: Paths,
    config: Any = None,
    *,
    release_dir: Path | str | None = None,
    s3_release: str | None = None,
    aws_profile: str | None = None,
    runner: AwsRunner | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> ImportSummary:
    """Verify and import a release into ``paths`` (see the module docstring).

    Args:
        paths: Destination run root.
        config: Loaded config; when given, every ``pairs_kept.csv`` drug must be canonical.
        release_dir: Local release directory (read only).
        s3_release: Release name to download from ``s3://g2m-data-v1/releases/`` first.
        aws_profile: Named AWS CLI profile for the download.
        runner, which: Test seams for the ``aws`` subprocess and the PATH lookup.

    Raises:
        ValueError: neither or both sources given.
        ReleaseError: verification failed, a required file or drug is missing.
        ContractViolation: the root holds different frozen splits.
    """
    if (release_dir is None) == (s3_release is None):
        raise ValueError("give exactly one of release_dir or s3_release")
    if s3_release is not None:
        source = download_s3_release(paths, s3_release, aws_profile=aws_profile, runner=runner, which=which)
    else:
        source = Path(release_dir)  # type: ignore[arg-type]
        if aws_profile is not None:
            raise ValueError("--aws-profile only applies to --s3-release")
    if not source.is_dir():
        raise ReleaseError(f"release directory not found: {source}")
    source = source.resolve()
    droplog = DropLog(STAGE)

    verified = verify_release(source)
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise ReleaseError(f"release {source} lacks {missing}")
    _check_frozen_splits(paths, source)
    n_pairs = _check_drugs(source, config)

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in RELEASE_FILES:
        src = source / name
        if not src.is_file():
            logger.info("[%s] %s not in the release; skipped", STAGE, name)
            continue
        if src.resolve() == (paths.processed_dir / name).resolve():
            copied.append(name)
            continue
        shutil.copyfile(src, paths.processed_dir / name)
        copied.append(name)
    for name in copied:
        if name in verified and sha256_file(paths.processed_dir / name) != verified[name]:
            raise ReleaseError(f"copy of {name} does not match SHA256SUMS")
    unverified = [name for name in copied if name not in verified and name != SHA256SUMS]
    if unverified:
        logger.warning("[%s] copied without a checksum (not listed in SHA256SUMS): %s", STAGE, unverified)

    known = pd.read_parquet(paths.known_amr, columns=["genome_id", "species"])
    qc = qc_table(known)
    write_parquet(qc, paths.qc)
    droplog.drop("QC not assessed (release ships no assemblies): every known-AMR genome marked qc_pass", 0, f"{len(qc)} genomes")

    release = _release_name(source)
    notes = list(NOTES)
    record = {
        "release": release,
        "source_dir": str(source),
        "s3_release": s3_release,
        "sha256_verified": True,
        "verified_files": sorted(verified),
        "copied_files": copied,
        "imported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
    }
    (paths.processed_dir / IMPORTED_RELEASE_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    droplog.write(paths.drop_log(STAGE.replace("-", "_")))
    logger.info(
        "[%s] imported release %s from %s: %d file(s), qc.parquet with %d genomes, %d kept pairs",
        STAGE, release, source, len(copied), len(qc), n_pairs,
    )
    return ImportSummary(
        release=release, source_dir=source, files=copied, n_genomes_qc=len(qc), n_pairs=n_pairs, notes=notes,
    )


def is_imported_release(paths: Paths) -> bool:
    """True when ``<root>/data/processed/IMPORTED_RELEASE.json`` exists."""
    return (paths.processed_dir / IMPORTED_RELEASE_FILE).is_file()


def read_imported_release(paths: Paths) -> dict[str, Any] | None:
    path = paths.processed_dir / IMPORTED_RELEASE_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


__all__: Sequence[str] = [
    "IMPORTED_RELEASE_FILE",
    "ImportSummary",
    "RELEASE_FILES",
    "REQUIRED_FILES",
    "ReleaseError",
    "S3_RELEASES_PREFIX",
    "download_s3_release",
    "is_imported_release",
    "parse_sha256sums",
    "qc_table",
    "read_imported_release",
    "run",
    "s3_release_uri",
    "sha256_file",
    "verify_release",
]
