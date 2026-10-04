"""Stage 0: pull the raw layer from a cloud bucket (or a local directory) into ``data/raw``.

The pipeline reads ``data/raw/ast_<source>.csv`` and ``data/raw/genomes/<genome_id>.fasta``
(``DATA_CONTRACT.md`` stages 1 and 3). Real data lives in a bucket; this module syncs a
bucket prefix into ``paths.raw_dir`` with the provider's own CLI and then checks the
layout. The provider is chosen from the URI scheme:

=========================================================  ==============================
URI                                                        Command
=========================================================  ==============================
``gs://bucket/prefix``                                     ``gcloud storage rsync -r``
                                                           (fallback ``gsutil -m rsync -r``)
``s3://bucket/prefix``                                     ``aws s3 sync``
``az://account/container/prefix``                          ``azcopy sync`` (rewritten to
                                                           the ``https://`` form below)
``https://account.blob.core.windows.net/container/prefix``  ``azcopy sync``
``/local/dir`` or ``file:///local/dir``                    pure-Python incremental copy
                                                           (no external tool; the testable path)
=========================================================  ==============================

**Credentials are never read, written or logged here.** The provider CLI finds them in
its own login state or environment (``gcloud auth login`` / VM service account,
``aws configure`` / instance role, ``azcopy login`` / ``AZCOPY_AUTO_LOGIN_TYPE``).
For S3, ``aws_profile`` (CLI ``--aws-profile NAME``) selects a named profile from
``~/.aws/config``: it sets ``AWS_PROFILE`` in the environment of the ``aws s3 sync``
subprocess only (this process's environment is untouched). The profile *name* is shown
in logs, the dry-run command and ``fetch_summary.json``; the environment's contents
(access keys, session tokens) never are.
Query strings (Azure SAS tokens) and ``user:password@`` parts are redacted from every
log line, from the dry-run output and from ``data/raw/fetch_summary.json``.

``include`` patterns select a subset of the prefix (``ast_*.csv``, ``genomes/*.fasta``).
They are glob patterns on the path *relative to the source prefix*; ``*`` matches ``/``
too (AWS semantics). Mapping per CLI:

* ``aws``:            ``--exclude '*' --include P1 --include P2``
* ``gcloud``/``gsutil``: one negative-lookahead regex ``--exclude='^(?!(?:P1|P2)).*'``
  -- those CLIs only have an exclude flag. Safe for bucket -> VM, where every
  object is a file (no directory pruning happens).
* ``azcopy``:         ``--include-pattern 'base(P1);base(P2)'`` -- azcopy matches the
  file *name*, so only the basename of each pattern is used.
* local:              ``fnmatch`` on the relative POSIX path.

Layout check after the sync (never fatal; recorded as ``problems`` in the returned
:class:`FetchSummary` and in ``data/raw/fetch_summary.json``):

* ``ast_bvbrc.csv`` or ``ast_ncbi.csv`` exists. Other ``ast_*.csv`` names are reported
  because ``ingest`` only reads those two.
* Every ``.fasta/.fa/.fna/.fas[.gz]`` under ``data/raw/genomes`` (recursively) is
  listed. When any of them is *not* ``data/raw/genomes/<genome_id>.fasta`` (other
  suffix, gzip, or a ``genomes/<genome_id>/<file>`` sub-folder) the files are left
  untouched and ``data/raw/genomes_manifest.csv`` (``genome_id, path, bytes``) is
  written instead of renaming anything. :func:`resolve_genome_paths` reads that
  manifest when present and falls back to ``genomes/*.fasta`` otherwise; stages that
  call ``paths.genome_fasta()`` today (qc, lineages, unitigs, the Snakefile) can adopt
  it without a contract change. Until they do, ``link_canonical=True`` adds
  ``genomes/<genome_id>.fasta`` symlinks for uncompressed non-canonical files so the
  existing consumers work unchanged.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit, urlunsplit

import pandas as pd

from genome2mic.errors import Genome2MicError, ToolNotAvailable
from genome2mic.paths import Paths

logger = logging.getLogger(__name__)

STAGE = "fetch"

MANIFEST_NAME = "genomes_manifest.csv"
"""``data/raw/genomes_manifest.csv`` -- written only when the genome layout is not canonical."""
SUMMARY_NAME = "fetch_summary.json"
"""``data/raw/fetch_summary.json`` -- redacted command + counts of the last real (non-dry) run."""
MANIFEST_COLUMNS: tuple[str, ...] = ("genome_id", "path", "bytes")

FASTA_SUFFIXES: tuple[str, ...] = (".fasta", ".fa", ".fna", ".fas")
CANONICAL_SUFFIX = ".fasta"
GZIP_SUFFIX = ".gz"
AST_GLOB = "ast_*.csv"
AST_READ_BY_INGEST: tuple[str, ...] = ("ast_bvbrc.csv", "ast_ncbi.csv")
AZURE_BLOB_HOST_SUFFIX = ".blob.core.windows.net"

PROVIDER_GCS = "gcs"
PROVIDER_S3 = "s3"
PROVIDER_AZURE = "azure"
PROVIDER_LOCAL = "local"

TOOLS: dict[str, tuple[str, ...]] = {
    PROVIDER_GCS: ("gcloud", "gsutil"),
    PROVIDER_S3: ("aws",),
    PROVIDER_AZURE: ("azcopy",),
}
"""Provider -> CLI executables in preference order."""

INSTALL_HINTS: dict[str, str] = {
    PROVIDER_GCS: (
        "Install the Google Cloud CLI (https://cloud.google.com/sdk/docs/install), then "
        "`gcloud auth login` or run on a VM with a service account that can read the bucket."
    ),
    PROVIDER_S3: (
        "Install the AWS CLI v2 (https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), "
        "then `aws configure` or attach an instance role that can read the bucket."
    ),
    PROVIDER_AZURE: (
        "Install azcopy (https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-v10), "
        "then `azcopy login` or set AZCOPY_AUTO_LOGIN_TYPE (e.g. MSI on an Azure VM)."
    ),
}

_REDACTED = "<redacted>"

AWS_PROFILE_ENV = "AWS_PROFILE"
AWS_ENV_CREDENTIAL_VARS: tuple[str, ...] = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
"""Environment credentials the AWS CLI uses *before* ``AWS_PROFILE`` (names only are ever logged)."""
_AWS_PROFILE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._@+\-]*$")


class FetchError(Genome2MicError):
    """The sync command ran but failed, or the source could not be read."""


# --------------------------------------------------------------------------- #
# source parsing + redaction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Source:
    """A parsed ``--source-uri``.

    Attributes:
        uri: The URI exactly as given (may carry a query string; never log it raw).
        provider: ``gcs`` / ``s3`` / ``azure`` / ``local``.
        location: What the CLI receives: the URI itself, the ``https://`` form for
            ``az://``, or the absolute local directory.
        local_path: The source directory for the local provider, else ``None``.
    """

    uri: str
    provider: str
    location: str
    local_path: Path | None = None

    @property
    def display(self) -> str:
        """Redacted form for logs and reports."""
        return redact_uri(self.location)


def redact_uri(uri: str) -> str:
    """Hide the query string (SAS tokens) and any ``user:password@`` part of ``uri``.

    Plain local paths come back unchanged unless they contain ``?`` or ``#``.
    """
    parts = urlsplit(uri)
    if not parts.scheme and not parts.netloc and not parts.query and not parts.fragment:
        return uri
    netloc = parts.netloc
    if "@" in netloc:
        netloc = f"{_REDACTED}@{netloc.rsplit('@', 1)[1]}"
    query = _REDACTED if parts.query else ""
    fragment = _REDACTED if parts.fragment else ""
    return urlunsplit((parts.scheme, netloc, parts.path, query, fragment))


def parse_source(uri: str) -> Source:
    """Classify ``uri`` by scheme and normalize it for the provider CLI.

    Raises:
        ValueError: unknown scheme, or an ``az://`` URI without ``account/container``.
        FileNotFoundError: a local source that is not an existing directory.
    """
    text = str(uri).strip()
    if not text:
        raise ValueError("source URI must not be empty")
    parts = urlsplit(text)
    scheme = parts.scheme.lower()

    if scheme == "gs":
        return Source(uri=text, provider=PROVIDER_GCS, location=text.rstrip("/"))
    if scheme == "s3":
        return Source(uri=text, provider=PROVIDER_S3, location=text.rstrip("/"))
    if scheme == "az":
        account = parts.netloc
        container, _, prefix = parts.path.lstrip("/").partition("/")
        if not account or not container:
            raise ValueError(
                f"az:// URIs must look like az://<account>/<container>[/<prefix>], got {redact_uri(text)!r}"
            )
        https = f"https://{account}{AZURE_BLOB_HOST_SUFFIX}/{container}"
        if prefix:
            https = f"{https}/{prefix.rstrip('/')}"
        if parts.query:
            https = f"{https}?{parts.query}"
        return Source(uri=text, provider=PROVIDER_AZURE, location=https)
    if scheme in ("http", "https") and parts.netloc.lower().endswith(AZURE_BLOB_HOST_SUFFIX):
        return Source(uri=text, provider=PROVIDER_AZURE, location=text)
    if scheme == "file":
        local = Path(unquote(parts.path))
    elif scheme == "" or (len(scheme) == 1 and os.name == "nt"):
        local = Path(text)
    else:
        raise ValueError(
            f"unsupported source URI scheme {scheme!r} in {redact_uri(text)!r}; expected gs://, s3://, az://, "
            f"https://<account>{AZURE_BLOB_HOST_SUFFIX}/..., file:// or a local directory"
        )
    local = local.expanduser().resolve()
    if not local.is_dir():
        raise FileNotFoundError(f"local source {local} is not a directory")
    return Source(uri=text, provider=PROVIDER_LOCAL, location=str(local), local_path=local)


# --------------------------------------------------------------------------- #
# command construction
# --------------------------------------------------------------------------- #


def select_tool(provider: str, *, dry_run: bool = False) -> str | None:
    """First executable of :data:`TOOLS` for ``provider`` found on ``PATH``.

    Returns ``None`` for the local provider. When nothing is installed: in dry-run mode
    the preferred tool name is returned so the command can still be shown; otherwise
    :class:`~genome2mic.errors.ToolNotAvailable` is raised with the install hint.
    """
    if provider == PROVIDER_LOCAL:
        return None
    candidates = TOOLS[provider]
    for tool in candidates:
        if shutil.which(tool):
            return tool
    if dry_run:
        logger.warning("none of %s is on PATH; showing the %s command anyway (dry run)", candidates, candidates[0])
        return candidates[0]
    raise ToolNotAvailable(candidates[0], hint=INSTALL_HINTS[provider])


def include_regex(include: Sequence[str]) -> str:
    """One Python regex that *excludes* everything not matching one of ``include``.

    gcloud/gsutil rsync have no include flag; their ``--exclude`` takes a Python regex
    matched against the path relative to the source. ``^(?!...)`` is a negative
    lookahead, so the regex matches (= excludes) exactly the paths that match none of
    the include globs. Commas are rejected because gcloud splits ``--exclude`` on them.
    """
    if not include:
        raise ValueError("include_regex needs at least one pattern")
    alternatives = []
    for pattern in include:
        if "," in pattern:
            raise ValueError(f"include pattern {pattern!r} contains a comma, which gcloud --exclude cannot carry")
        alternatives.append(fnmatch.translate(pattern))
    return "^(?!(?:" + "|".join(alternatives) + ")).*"


def build_command(source: Source, dest: Path, include: Sequence[str] | None, tool: str) -> list[str]:
    """The argv the provider CLI runs for ``source`` -> ``dest`` (no shell involved).

    ``include`` is mapped to the CLI's own filter flags as described in the module
    docstring. Raises ``ValueError`` for the local provider (there is no command).
    """
    patterns = [p for p in (include or []) if p]
    target = str(dest)
    if source.provider == PROVIDER_GCS:
        if tool == "gcloud":
            cmd = ["gcloud", "storage", "rsync", "-r", source.location, target]
            if patterns:
                cmd.append(f"--exclude={include_regex(patterns)}")
            return cmd
        if tool == "gsutil":
            cmd = ["gsutil", "-m", "rsync", "-r"]
            if patterns:
                cmd += ["-x", include_regex(patterns)]
            return cmd + [source.location, target]
        raise ValueError(f"unknown GCS tool {tool!r}")
    if source.provider == PROVIDER_S3:
        cmd = ["aws", "s3", "sync", source.location, target]
        if patterns:
            cmd += ["--exclude", "*"]
            for pattern in patterns:
                cmd += ["--include", pattern]
        return cmd
    if source.provider == PROVIDER_AZURE:
        cmd = ["azcopy", "sync", source.location, target, "--recursive=true"]
        if patterns:
            names = ";".join(dict.fromkeys(PurePosixPath(p).name for p in patterns))
            cmd += ["--include-pattern", names]
        return cmd
    raise ValueError("the local provider copies in-process; there is no command to build")


def validate_aws_profile(profile: str) -> str:
    """Return ``profile`` if it is a plausible AWS CLI profile name, else raise ``ValueError``.

    Accepts letters, digits and ``. _ @ + -`` (not as the first character for ``. @ + -``),
    so a typo such as ``--aws-profile "two words"`` or a pasted ``KEY=value`` fails early
    instead of reaching the CLI. The value is passed through the environment, never a
    shell, so this is a usability check, not an injection guard.
    """
    if not isinstance(profile, str) or not _AWS_PROFILE_RE.fullmatch(profile):
        raise ValueError(
            f"invalid AWS profile name {profile!r}: use the name of a profile in ~/.aws/config "
            "(letters, digits, '.', '_', '@', '+', '-')"
        )
    return profile


def subprocess_env(aws_profile: str | None) -> dict[str, str] | None:
    """Environment for the provider CLI: ``None`` (inherit) or a copy with ``AWS_PROFILE`` set."""
    if aws_profile is None:
        return None
    return {**os.environ, AWS_PROFILE_ENV: aws_profile}


def redact_command(command: Sequence[str]) -> list[str]:
    """Copy of ``command`` with every URI-looking argument passed through :func:`redact_uri`."""
    return [redact_uri(arg) if "://" in arg else arg for arg in command]


# --------------------------------------------------------------------------- #
# local copy backend (the testable path)
# --------------------------------------------------------------------------- #


def matches_include(relative_posix: str, include: Sequence[str] | None) -> bool:
    """``True`` when ``relative_posix`` matches any include glob (or there are none).

    Uses :func:`fnmatch.fnmatchcase`, where ``*`` also matches ``/`` -- the same rule
    ``aws s3 sync --include`` applies.
    """
    if not include:
        return True
    return any(fnmatch.fnmatchcase(relative_posix, pattern) for pattern in include if pattern)


@dataclass
class LocalCopyResult:
    """Counts from :func:`sync_local`."""

    n_selected: int = 0
    n_copied: int = 0
    n_skipped: int = 0
    bytes_selected: int = 0
    bytes_copied: int = 0


def sync_local(src: Path, dest: Path, include: Sequence[str] | None, *, dry_run: bool) -> LocalCopyResult:
    """Incremental copy of ``src`` into ``dest`` (like ``rsync -a`` without deletes).

    A destination file with the same size and modification time (to the second) is
    skipped, so an interrupted copy resumes. Nothing is written when ``dry_run``.
    """
    result = LocalCopyResult()
    for dirpath, _dirnames, filenames in os.walk(src):
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if not path.is_file():
                continue
            rel = path.relative_to(src)
            if not matches_include(rel.as_posix(), include):
                continue
            size = path.stat().st_size
            result.n_selected += 1
            result.bytes_selected += size
            target = dest / rel
            if _same_file_state(path, target):
                result.n_skipped += 1
                continue
            result.n_copied += 1
            result.bytes_copied += size
            if dry_run:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return result


def _same_file_state(src: Path, dest: Path) -> bool:
    if not dest.is_file():
        return False
    a, b = src.stat(), dest.stat()
    return a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime)


# --------------------------------------------------------------------------- #
# layout check: AST files, genomes, manifest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GenomeFile:
    """One FASTA-like file found under ``data/raw/genomes``."""

    genome_id: str
    path: Path
    bytes: int
    canonical: bool
    """``True`` for ``data/raw/genomes/<genome_id>.fasta`` (uncompressed, top level)."""


def strip_fasta_suffix(name: str) -> str | None:
    """``'573.2002.fna.gz'`` -> ``'573.2002'``; ``None`` when ``name`` is not FASTA-like."""
    stem = name
    if stem.endswith(GZIP_SUFFIX):
        stem = stem[: -len(GZIP_SUFFIX)]
    for suffix in FASTA_SUFFIXES:
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return stem[: -len(suffix)]
    return None


def scan_genomes(genomes_dir: Path, problems: list[str] | None = None) -> list[GenomeFile]:
    """Every FASTA-like file under ``genomes_dir`` (recursive) with its ``genome_id``.

    ``genome_id`` rules: a top-level file uses its name minus the FASTA (and ``.gz``)
    suffix; a file inside a sub-folder uses the first folder name
    (``genomes/573.2002/anything.fna`` -> ``573.2002``) unless that folder holds
    several FASTA files, in which case each file's own stem is used and a problem is
    recorded. Duplicate ids keep the canonical/first entry; a symlink that resolves to
    the same file is not a duplicate. Results are sorted by ``genome_id``.
    """
    issues = problems if problems is not None else []
    if not genomes_dir.is_dir():
        return []
    per_folder: dict[str, list[Path]] = {}
    top_level: list[Path] = []
    for path in sorted(genomes_dir.rglob("*")):
        if not path.is_file() or strip_fasta_suffix(path.name) is None:
            continue
        rel = path.relative_to(genomes_dir)
        if len(rel.parts) == 1:
            top_level.append(path)
        else:
            per_folder.setdefault(rel.parts[0], []).append(path)

    found: dict[str, GenomeFile] = {}

    def add(gid: str, path: Path, canonical: bool) -> None:
        entry = GenomeFile(gid, path, path.stat().st_size, canonical)
        previous = found.get(gid)
        if previous is None:
            found[gid] = entry
            return
        if os.path.samefile(previous.path, path):
            if canonical and not previous.canonical:
                found[gid] = entry
            return
        issues.append(
            f"duplicate genome_id {gid!r}: {previous.path.relative_to(genomes_dir)} and "
            f"{path.relative_to(genomes_dir)} (kept the first)"
        )

    for path in top_level:
        gid = strip_fasta_suffix(path.name) or path.name
        canonical = path.suffix == CANONICAL_SUFFIX
        add(gid, path, canonical)
    for folder, files in per_folder.items():
        if len(files) == 1:
            add(folder, files[0], False)
            continue
        issues.append(
            f"genomes/{folder}/ holds {len(files)} FASTA files; using each file's stem as genome_id"
        )
        for path in files:
            add(strip_fasta_suffix(path.name) or path.name, path, False)
    return [found[gid] for gid in sorted(found)]


def write_manifest(genomes: Iterable[GenomeFile], paths: Paths) -> Path:
    """Write ``data/raw/genomes_manifest.csv`` with paths relative to ``paths.root``."""
    rows = []
    for g in genomes:
        try:
            rel = g.path.resolve().relative_to(paths.root.resolve()).as_posix()
        except ValueError:
            rel = str(g.path.resolve())
        rows.append({"genome_id": g.genome_id, "path": rel, "bytes": int(g.bytes)})
    frame = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    target = paths.raw_dir / MANIFEST_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    logger.info("[%s] wrote %d genome paths to %s", STAGE, len(frame), target)
    return target


def manifest_path(paths: Paths) -> Path:
    """``data/raw/genomes_manifest.csv``."""
    return paths.raw_dir / MANIFEST_NAME


def resolve_genome_paths(paths: Paths) -> dict[str, Path]:
    """``genome_id -> FASTA path`` for every genome on disk.

    Reads ``data/raw/genomes_manifest.csv`` when it exists (relative paths are resolved
    against ``paths.root``); otherwise lists ``data/raw/genomes/*.fasta`` exactly like
    ``qc.list_genome_fastas``. This is the helper the per-genome stages should call
    instead of ``paths.genome_fasta(gid)`` once they adopt non-canonical layouts.
    """
    manifest = manifest_path(paths)
    if manifest.is_file():
        table = pd.read_csv(manifest, dtype={"genome_id": str, "path": str})
        missing = [c for c in MANIFEST_COLUMNS if c not in table.columns]
        if missing:
            raise FetchError(f"{manifest} lacks columns {missing}; expected {list(MANIFEST_COLUMNS)}")
        out: dict[str, Path] = {}
        for gid, rel in zip(table["genome_id"], table["path"], strict=True):
            p = Path(str(rel))
            out[str(gid)] = p if p.is_absolute() else paths.root / p
        return out
    if not paths.genomes_dir.is_dir():
        return {}
    return {p.stem: p for p in sorted(paths.genomes_dir.glob(f"*{CANONICAL_SUFFIX}")) if p.is_file()}


def link_canonical_names(genomes: Iterable[GenomeFile], paths: Paths, problems: list[str]) -> int:
    """Add ``genomes/<genome_id>.fasta`` symlinks for uncompressed non-canonical files.

    Never overwrites a real file. Gzip-compressed genomes are skipped (a ``.fasta``
    name would hide the compression from readers). Returns the number of links made.
    """
    made = 0
    for g in genomes:
        if g.canonical:
            continue
        if g.path.name.endswith(GZIP_SUFFIX):
            problems.append(f"{g.genome_id}: gzip-compressed genome not linked ({g.path.name}); decompress it")
            continue
        link = paths.genomes_dir / f"{g.genome_id}{CANONICAL_SUFFIX}"
        if link.is_symlink():
            if os.path.realpath(link) == os.path.realpath(g.path):
                continue
            link.unlink()
        elif link.exists():
            problems.append(f"{g.genome_id}: {link.name} already exists and is not a symlink; not replaced")
            continue
        link.symlink_to(os.path.relpath(g.path, link.parent))
        made += 1
    return made


def list_ast_files(paths: Paths) -> list[Path]:
    """``data/raw/ast_*.csv`` (top level only)."""
    if not paths.raw_dir.is_dir():
        return []
    return sorted(p for p in paths.raw_dir.glob(AST_GLOB) if p.is_file())


def directory_size(root: Path) -> tuple[int, int]:
    """``(n_files, total_bytes)`` under ``root`` (symlinks not followed)."""
    n_files = total = 0
    if not root.is_dir():
        return 0, 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            try:
                total += path.stat().st_size
            except OSError:
                continue
            n_files += 1
    return n_files, total


# --------------------------------------------------------------------------- #
# summary + stage entry point
# --------------------------------------------------------------------------- #


@dataclass
class FetchSummary:
    """What :func:`run` did. Every URI/command field is already redacted."""

    source: str
    provider: str
    tool: str | None
    command: list[str]
    dest: str
    dry_run: bool
    n_files: int = 0
    total_bytes: int = 0
    n_genomes: int = 0
    n_ast_files: int = 0
    ast_files: list[str] = field(default_factory=list)
    manifest: str | None = None
    n_links: int = 0
    problems: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    aws_profile: str | None = None
    """``AWS_PROFILE`` set for the ``aws`` subprocess (a profile name, not a credential)."""

    @property
    def layout_ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def command_line(self) -> str:
        """Shell-quoted (redacted) command, or the local copy description.

        With an AWS profile the line starts with ``AWS_PROFILE=<name>`` so the printed
        command reproduces the run when pasted into a shell.
        """
        line = shlex.join(self.command)
        if self.aws_profile:
            line = f"{AWS_PROFILE_ENV}={shlex.quote(self.aws_profile)} {line}"
        return line


def run(
    paths: Paths,
    source_uri: str,
    include: Sequence[str] | None = None,
    dry_run: bool = False,
    *,
    link_canonical: bool = False,
    aws_profile: str | None = None,
) -> FetchSummary:
    """Sync ``source_uri`` into ``paths.raw_dir`` and validate the raw layout.

    Args:
        paths: Project paths; the destination is ``paths.raw_dir``.
        source_uri: ``gs://``, ``s3://``, ``az://``, Azure ``https://`` blob URL,
            ``file://`` or a local directory (see the module docstring).
        include: Optional glob patterns relative to the source prefix
            (``['ast_*.csv', 'genomes/*.fasta']``). ``None``/empty = everything.
        dry_run: Build and report the command, write nothing, run nothing.
        link_canonical: After the sync, add ``genomes/<genome_id>.fasta`` symlinks for
            uncompressed genomes that have another suffix or live in sub-folders.
        aws_profile: Named AWS CLI profile for an ``s3://`` source. Set as
            ``AWS_PROFILE`` for the ``aws s3 sync`` subprocess only; ``None`` inherits
            the current environment (an exported ``AWS_PROFILE`` still applies).

    Returns:
        :class:`FetchSummary`. Also writes ``data/raw/fetch_summary.json`` and, when
        the genome layout is not canonical, ``data/raw/genomes_manifest.csv``
        (neither in dry-run mode).

    Raises:
        ValueError: unknown URI scheme; ``aws_profile`` given for a non-S3 source or
            not a valid profile name.
        ToolNotAvailable: the provider CLI is not on ``PATH`` (non-dry runs).
        FetchError: the CLI exited non-zero.
    """
    started = time.perf_counter()
    source = parse_source(source_uri)
    if aws_profile is not None:
        validate_aws_profile(aws_profile)
        if source.provider != PROVIDER_S3:
            raise ValueError(
                f"--aws-profile only applies to s3:// sources; {source.display!r} is a {source.provider} source"
            )
    patterns = [p for p in (include or []) if p]
    dest = paths.raw_dir
    tool = select_tool(source.provider, dry_run=dry_run)

    if source.provider == PROVIDER_LOCAL:
        assert source.local_path is not None
        copy = sync_local(source.local_path, dest, patterns, dry_run=dry_run)
        command = [
            "<python copy>", source.location, str(dest),
            *(f"--include={p}" for p in patterns),
        ]
        verb = "would copy" if dry_run else "copied"
        logger.info(
            "[%s] local source %s -> %s: %d files selected (%s), %s %d (%s), %d already up to date",
            STAGE, source.display, dest, copy.n_selected, _human(copy.bytes_selected),
            verb, copy.n_copied, _human(copy.bytes_copied), copy.n_skipped,
        )
    else:
        assert tool is not None
        command = build_command(source, dest, patterns, tool)
        shown = shlex.join(redact_command(command))
        if aws_profile is not None:
            # only the profile name: the environment itself is never logged
            shown = f"{AWS_PROFILE_ENV}={shlex.quote(aws_profile)} {shown}"
            shadowing = [name for name in AWS_ENV_CREDENTIAL_VARS if os.environ.get(name)]
            if shadowing:
                logger.warning(
                    "[%s] %s set in the environment; the AWS CLI uses environment credentials before "
                    "%s=%s. Unset them to use the profile.",
                    STAGE, " and ".join(shadowing), AWS_PROFILE_ENV, aws_profile,
                )
        if dry_run:
            logger.info("[%s] dry run; would execute: %s", STAGE, shown)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            logger.info("[%s] executing: %s", STAGE, shown)
            completed = subprocess.run(  # noqa: S603 -- argv list, no shell
                command, check=False, env=subprocess_env(aws_profile)
            )
            if completed.returncode != 0:
                raise FetchError(f"{tool} exited with status {completed.returncode}: {shown}")

    summary = FetchSummary(
        source=source.display,
        provider=source.provider,
        tool=tool,
        command=redact_command(command),
        dest=str(dest),
        dry_run=dry_run,
        aws_profile=aws_profile,
    )
    if dry_run:
        summary.elapsed_s = time.perf_counter() - started
        return summary

    _validate_layout(paths, summary, link_canonical=link_canonical)
    summary.elapsed_s = time.perf_counter() - started
    _write_summary(paths, summary)
    logger.info(
        "[%s] %s: %d files, %s under %s; %d genomes, %d AST file(s); %d layout problem(s)",
        STAGE, source.provider, summary.n_files, _human(summary.total_bytes), dest,
        summary.n_genomes, summary.n_ast_files, len(summary.problems),
    )
    for problem in summary.problems:
        logger.warning("[%s] layout: %s", STAGE, problem)
    return summary


def _validate_layout(paths: Paths, summary: FetchSummary, *, link_canonical: bool) -> None:
    problems = summary.problems
    summary.n_files, summary.total_bytes = directory_size(paths.raw_dir)

    ast_files = list_ast_files(paths)
    summary.ast_files = [p.name for p in ast_files]
    summary.n_ast_files = len(ast_files)
    if not ast_files:
        problems.append(f"no {AST_GLOB} under {paths.raw_dir} (ingest needs ast_bvbrc.csv and/or ast_ncbi.csv)")
    elif not any(p.name in AST_READ_BY_INGEST for p in ast_files):
        problems.append(
            f"AST files {summary.ast_files} are not named {list(AST_READ_BY_INGEST)}; ingest reads only those names"
        )

    genomes = scan_genomes(paths.genomes_dir, problems)
    summary.n_genomes = len(genomes)
    if not genomes:
        problems.append(f"no FASTA files under {paths.genomes_dir}")
        return
    non_canonical = [g for g in genomes if not g.canonical]
    if non_canonical:
        manifest = write_manifest(genomes, paths)
        summary.manifest = str(manifest)
        example = non_canonical[0].path.relative_to(paths.genomes_dir)
        problems.append(
            f"{len(non_canonical)}/{len(genomes)} genomes are not genomes/<genome_id>.fasta (e.g. {example}); "
            f"wrote {manifest.name} -- stages must use fetch.resolve_genome_paths() or run with --link-canonical"
        )
        if link_canonical:
            summary.n_links = link_canonical_names(non_canonical, paths, problems)
            logger.info("[%s] created %d canonical .fasta symlinks", STAGE, summary.n_links)
    elif manifest_path(paths).is_file():
        problems.append(f"{MANIFEST_NAME} exists but every genome is canonical; delete it if stale")


def _write_summary(paths: Paths, summary: FetchSummary) -> Path:
    target = paths.raw_dir / SUMMARY_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {**summary.to_dict(), "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _human(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n_bytes} B"


__all__ = [
    "AWS_PROFILE_ENV",
    "AST_GLOB",
    "AST_READ_BY_INGEST",
    "FASTA_SUFFIXES",
    "FetchError",
    "FetchSummary",
    "GenomeFile",
    "INSTALL_HINTS",
    "LocalCopyResult",
    "MANIFEST_COLUMNS",
    "MANIFEST_NAME",
    "STAGE",
    "SUMMARY_NAME",
    "Source",
    "TOOLS",
    "build_command",
    "directory_size",
    "include_regex",
    "link_canonical_names",
    "list_ast_files",
    "manifest_path",
    "matches_include",
    "parse_source",
    "redact_command",
    "redact_uri",
    "resolve_genome_paths",
    "run",
    "scan_genomes",
    "select_tool",
    "strip_fasta_suffix",
    "subprocess_env",
    "sync_local",
    "validate_aws_profile",
    "write_manifest",
]
