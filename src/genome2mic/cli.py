"""Command-line entry point: ``python -m genome2mic <stage> [--root ..] [--configs-dir ..]``.

Subcommands (``.context/DESIGN.md``): ``synth``, ``fetch``, ``import-release``,
``ingest``, ``qc``, ``known-amr``, ``lineages``, ``splits``, ``unitigs``, ``train``,
``evaluate``, ``report``, ``compare-oof``, ``predict``, ``run-all``. Each builds a :class:`genome2mic.paths.Paths`
from ``--root``/``--configs-dir`` and calls the stage's ``run(paths, config, **opts)``.

``fetch`` syncs the raw layer (AST exports + genomes) from a cloud bucket or a local
directory into ``<root>/data/raw`` (:mod:`genome2mic.ingest.fetch`); ``run-all
--source-uri ...`` runs it first and skips ``synth`` (real data replaces the
synthetic layer).

``--configs-dir`` defaults to ``<root>/configs`` when that directory exists (the
synthetic run writes its size-adjusted copy there), else ``./configs``. ``synth``
reads the *source* configs (default ``./configs``) and refuses to overwrite them.

Every stage logs the counts of what it dropped (``DropLog``); the CLI only prints a
one-line summary per stage and, for ``predict``, the report JSON. Reports carry the
disclaimer: these are predictions of in-vitro susceptibility, not prescribing advice.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from genome2mic.paths import Paths

logger = logging.getLogger(__name__)

PIPELINE_STAGES: tuple[str, ...] = (
    "ingest", "qc", "known-amr", "lineages", "splits", "unitigs", "train", "evaluate", "report",
)
"""Order of the stages ``run-all`` executes after ``synth``."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _resolve_paths(args: argparse.Namespace, *, for_synth: bool = False) -> Paths:
    root = Path(args.root)
    configs = args.configs_dir
    if configs is None:
        if for_synth:
            configs = Path("configs")
        else:
            candidate = root / "configs"
            configs = candidate if candidate.is_dir() else Path("configs")
    return Paths(root=root, configs_dir=Path(configs))


def _load_config(paths: Paths) -> Any:
    from genome2mic.config import load_config  # noqa: PLC0415

    return load_config(paths.configs_dir)


def _timed(name: str, fn: Callable[[], Any]) -> Any:
    started = time.perf_counter()
    logger.info("[%s] start", name)
    result = fn()
    elapsed = time.perf_counter() - started
    logger.info("[%s] done in %.1fs", name, elapsed)
    print(f"{name}: done in {elapsed:.1f}s")
    return result


# --------------------------------------------------------------------------- #
# stage commands
# --------------------------------------------------------------------------- #


def cmd_synth(args: argparse.Namespace) -> int:
    from genome2mic.synthetic import generate  # noqa: PLC0415

    paths = _resolve_paths(args, for_synth=True)
    summary = _timed(
        "synth",
        lambda: generate.run(paths, seed=args.synth_seed, n_kpneu=args.n_kpneu, n_ecoli=args.n_ecoli, genome_length=args.genome_length),
    )
    print(f"synthetic data written under {paths.root} (seed {args.synth_seed}; {summary.get('n_genomes', '?')} genomes). "
          "SYNTHETIC DATA: do not present any metric from it as real.")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from genome2mic.ingest import fetch  # noqa: PLC0415

    paths = _resolve_paths(args)
    summary = _timed(
        "fetch",
        lambda: fetch.run(
            paths,
            source_uri=args.source_uri,
            include=args.include or None,
            dry_run=args.dry_run,
            link_canonical=getattr(args, "link_canonical", False),
            aws_profile=getattr(args, "aws_profile", None),
        ),
    )
    if summary.dry_run:
        print(f"fetch (dry run): would run -> {summary.command_line()}")
        print(f"fetch (dry run): destination {summary.dest}; nothing was written")
        return 0
    print(
        f"fetch: {summary.provider} -> {summary.dest}: {summary.n_files} files, {summary.total_bytes} bytes; "
        f"{summary.n_genomes} genomes, {summary.n_ast_files} AST file(s) {summary.ast_files}"
    )
    if summary.manifest:
        print(f"fetch: genome layout is not <genome_id>.fasta; manifest -> {summary.manifest} ({summary.n_links} symlinks made)")
    for problem in summary.problems:
        print(f"fetch: LAYOUT WARNING: {problem}")
    return 0


def cmd_import_release(args: argparse.Namespace) -> int:
    from genome2mic.ingest import release  # noqa: PLC0415

    paths = _resolve_paths(args)
    config = _load_config(paths)
    summary = _timed(
        "import-release",
        lambda: release.run(
            paths,
            config,
            release_dir=Path(args.release_dir) if args.release_dir else None,
            s3_release=args.s3_release,
            aws_profile=args.aws_profile,
        ),
    )
    print(
        f"import-release: {summary.release or '(unnamed release)'} from {summary.source_dir} -> {paths.processed_dir}: "
        f"{len(summary.files)} file(s), sha256 verified; qc.parquet with {summary.n_genomes_qc} genomes; "
        f"{summary.n_pairs} kept pairs"
    )
    for note in summary.notes:
        print(f"import-release: note: {note}")
    return 0


def cmd_release_feature_spec(args: argparse.Namespace) -> int:
    from genome2mic.predict import release_features  # noqa: PLC0415

    paths = _resolve_paths(args)
    config = _load_config(paths)
    written = release_features.write_specs_for_root(
        paths, config, Path(args.amrfinder_db) if args.amrfinder_db else None
    )
    for path in written:
        print(f"release-feature-spec: wrote {path}")
    print(f"release-feature-spec: {paths.models_manifest} feature_naming = {release_features.FEATURE_NAMING_NCBI_RELEASE}")
    return 0


def cmd_compare_oof(args: argparse.Namespace) -> int:
    from genome2mic.eval import oof_compare  # noqa: PLC0415

    paths = _resolve_paths(args)
    config = _load_config(paths)
    result = _timed(
        "compare-oof",
        lambda: oof_compare.run(
            paths,
            config,
            develop_preds=Path(args.develop_preds),
            out_dir=Path(args.out),
            breakpoints_csv=Path(args.breakpoints) if args.breakpoints else None,
            amrfinder_db=Path(args.amrfinder_db) if getattr(args, "amrfinder_db", None) else None,
        ),
    )
    print(f"compare-oof: {len(result.table)} rows -> {result.csv_path} and {result.md_path}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from genome2mic import ingest  # noqa: PLC0415

    paths = _resolve_paths(args)
    config = _load_config(paths)
    labels = _timed("ingest", lambda: ingest.run(paths, config, min_ns=args.min_ns, min_s=args.min_s, min_levels=args.min_levels))
    print(f"labels: {len(labels)} rows -> {paths.labels}; pairs kept -> {paths.pairs_kept}")
    return 0


def cmd_qc(args: argparse.Namespace) -> int:
    from genome2mic import qc  # noqa: PLC0415

    paths = _resolve_paths(args)
    config = _load_config(paths)
    table = _timed("qc", lambda: qc.run(paths, config))
    n_pass = int(table["qc_pass"].sum()) if "qc_pass" in table.columns else 0
    print(f"qc: {n_pass}/{len(table)} genomes pass -> {paths.qc}")
    return 0


def cmd_known_amr(args: argparse.Namespace) -> int:
    from genome2mic.features import known_amr  # noqa: PLC0415

    paths = _resolve_paths(args)
    config = _load_config(paths)
    table = _timed("known-amr", lambda: known_amr.run(paths, config))
    print(f"known-amr: {len(table)} genomes x {max(0, table.shape[1] - 2)} feature columns -> {paths.known_amr}")
    return 0


def cmd_lineages(args: argparse.Namespace) -> int:
    from genome2mic.splits import lineages  # noqa: PLC0415

    paths = _resolve_paths(args)
    config = _load_config(paths)
    table = _timed(
        "lineages",
        lambda: lineages.run(paths, config, threshold=args.threshold, backend=args.lineage_backend, sketch_size=args.sketch_size),
    )
    print(f"lineages: {len(table)} genomes in {table['lineage_cluster'].nunique()} clusters -> {paths.lineages}")
    return 0


def cmd_splits(args: argparse.Namespace) -> int:
    from genome2mic.splits import make_splits  # noqa: PLC0415

    paths = _resolve_paths(args)
    config = _load_config(paths)
    lo, hi = args.test_range
    table = _timed(
        "splits",
        lambda: make_splits.run(paths, config, seed=args.split_seed, force=args.force, test_range=(lo, hi), n_folds=args.n_folds),
    )
    n_test = int((table["split"] == "test").sum())
    print(f"splits: {len(table)} genomes, {n_test} test ({n_test / max(len(table), 1):.1%}) -> {paths.splits} (frozen)")
    return 0


def cmd_unitigs(args: argparse.Namespace) -> int:
    from genome2mic.features import unitigs  # noqa: PLC0415

    paths = _resolve_paths(args)
    config = _load_config(paths)
    species = [s.upper() for s in args.species] if getattr(args, "species", None) else None
    max_kmer_genomes = getattr(args, "unitig_max_kmer_genomes", unitigs.DEFAULT_MAX_KMER_GENOMES)
    threads = getattr(args, "unitig_threads", None)
    table = _timed(
        "unitigs",
        lambda: unitigs.run(
            paths, config, species=species, backend=args.unitig_backend,
            max_kmer_genomes=max_kmer_genomes, threads=threads,
        ),
    )
    print(table.to_string(index=False))
    return 0


def _train_config(args: argparse.Namespace) -> Any:
    from genome2mic.models.train import DEFAULT_MODELS, TrainConfig  # noqa: PLC0415

    return TrainConfig(
        models=tuple(args.models) if args.models else DEFAULT_MODELS,
        top_k=args.top_k,
        min_count=args.min_count,
        nthread=args.nthread,
        seed=args.train_seed,
        max_rounds=args.max_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        lolo=not args.no_lolo and not getattr(args, "cv_only", False),
        ablation=args.ablation,
        cv_only=getattr(args, "cv_only", False),
        workers=max(1, int(getattr(args, "workers", 1) or 1)),
        band=getattr(args, "band", "asym_tuned") or "asym_tuned",
        exact_weight=float(getattr(args, "exact_weight", 2.0)),
        model_select=not getattr(args, "no_model_select", False),
    )


def cmd_train(args: argparse.Namespace) -> int:
    """Train every kept pair, or with ``--species``/``--drugs`` only the matching kept pairs (a shard).

    A shard merges its pairs into ``models/manifest.json`` and ``drop_log_train.csv``
    instead of replacing them, so shards can run as separate jobs.
    """
    from genome2mic.models import train  # noqa: PLC0415

    paths = _resolve_paths(args)
    config = _load_config(paths)
    species = getattr(args, "train_species", None)
    drugs = getattr(args, "train_drugs", None)
    pairs = train.select_pairs(paths, config, species=species, drugs=drugs) if (species or drugs) else None
    if pairs is not None:
        print(f"train: {len(pairs)} pair(s) selected: {', '.join(f'{s} x {d}' for s, d in pairs)}")
    db = getattr(args, "amrfinder_db", None)
    summary = _timed("train", lambda: train.run(
        paths, config, train_config=_train_config(args), pairs=pairs, amrfinder_db=Path(db) if db else None,
    ))
    print(summary.to_string(index=False))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from genome2mic.eval import run as eval_run  # noqa: PLC0415

    paths = _resolve_paths(args)
    config = _load_config(paths)
    table = _timed("evaluate", lambda: eval_run.run(paths, config))
    test = table.loc[table["split"] == "test"]
    # EA is computed on exact lab MICs only, so its denominator n_exact sits next to it;
    # the re-derived VME (lab MIC under the call breakpoint) follows the as-reported block.
    cols = [
        "species", "drug", "model", "vme_rate", "call_vme_rate_rederived", "active_call_rate_s_rederived",
        "me_rate", "categorical_agreement", "essential_agreement", "n_exact", "vme_rate_rederived", "n",
    ]
    if test.empty:
        cv = table.loc[table["split"] == "cv"]
        print("Test set not scored (cv-only run). Out-of-fold CV metrics (VME first):")
        print(cv[[c for c in cols if c in cv.columns]].to_string(index=False))
        return 0
    cols = [c for c in cols if c in test.columns]
    print("Test-set metrics (VME first; synthetic data if data/raw/SYNTHETIC_DATA.md exists):")
    print(test[cols].to_string(index=False))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from genome2mic.eval import report  # noqa: PLC0415

    paths = _resolve_paths(args)
    config = _load_config(paths)
    out = _timed("report", lambda: report.run(paths, config, make_figures=not args.no_figures))
    print(f"report: {out.report_path} ({len(out.figures)} figures)")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from genome2mic.predict.pipeline import PredictionPipeline  # noqa: PLC0415

    paths = _resolve_paths(args)
    models_dir = Path(args.models_dir) if args.models_dir else paths.models_dir
    pipeline = PredictionPipeline(
        models_dir, paths.configs_dir, amrfinder_tsv=Path(args.amrfinder_tsv) if args.amrfinder_tsv else None
    )
    pipeline.load()
    sample_id = args.sample_id or Path(args.fasta).stem
    report = pipeline.run(Path(args.fasta), sample_id)
    print(json.dumps(report, indent=2, default=_json_default))
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    """``synth`` (unless ``--no-synth``) then every pipeline stage in order."""
    started = time.perf_counter()
    timings: list[tuple[str, float]] = []

    def step(name: str, fn: Callable[[argparse.Namespace], int], ns: argparse.Namespace) -> None:
        t0 = time.perf_counter()
        code = fn(ns)
        if code:
            raise SystemExit(code)
        timings.append((name, time.perf_counter() - t0))

    base = dict(vars(args))
    source_uri = getattr(args, "source_uri", None)
    if source_uri:
        # Real data: fetch replaces the synthetic layer, so synth is skipped.
        if not args.no_synth:
            logger.info("--source-uri given: skipping synth (fetched real data must not be overwritten)")
        fetch_ns = argparse.Namespace(**{**base, "dry_run": False})
        step("fetch", cmd_fetch, fetch_ns)
    elif not args.no_synth:
        synth_ns = argparse.Namespace(**{**base, "configs_dir": args.configs_dir})
        step("synth", cmd_synth, synth_ns)
    # After synth the size-adjusted configs live under <root>/configs.
    stage_ns = argparse.Namespace(**{**base, "configs_dir": args.configs_dir})
    if stage_ns.configs_dir is None and (Path(args.root) / "configs").is_dir():
        stage_ns.configs_dir = Path(args.root) / "configs"
    stage_ns.force = args.force_splits
    stage_ns.species = None
    for name, fn in (
        ("ingest", cmd_ingest), ("qc", cmd_qc), ("known-amr", cmd_known_amr), ("lineages", cmd_lineages),
        ("splits", cmd_splits), ("unitigs", cmd_unitigs), ("train", cmd_train), ("evaluate", cmd_evaluate),
        ("report", cmd_report),
    ):
        step(name, fn, stage_ns)
    total = time.perf_counter() - started
    print("\nrun-all wall time per stage:")
    for name, seconds in timings:
        print(f"  {name:<10s} {seconds:8.1f}s")
    print(f"  {'total':<10s} {total:8.1f}s")
    return 0


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=".", help="project root containing data/, results/, models/ (default: .)")
    parser.add_argument(
        "--configs-dir", default=None,
        help="configs directory (default: <root>/configs if it exists, else ./configs; synth: ./configs)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    parser.add_argument("-q", "--quiet", action="store_true", help="WARNING logging only")


def _add_synth_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", dest="synth_seed", type=int, default=7, help="synthetic data generator seed")
    parser.add_argument("--n-kpneu", type=int, default=700)
    parser.add_argument("--n-ecoli", type=int, default=300)
    parser.add_argument("--genome-length", type=int, default=60_000)


def _add_fetch_args(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "--source-uri", required=required, default=None,
        help="bucket prefix or directory holding ast_*.csv and genomes/: gs://, s3://, az://<account>/<container>/<prefix>, "
             "https://<account>.blob.core.windows.net/..., file:// or a local path",
    )
    parser.add_argument(
        "--include", action="append", default=[], metavar="GLOB",
        help="only sync paths matching this glob (relative to --source-uri; '*' also matches '/'); repeatable, "
             "e.g. --include 'ast_*.csv' --include 'genomes/*.fasta'",
    )
    parser.add_argument(
        "--link-canonical", action="store_true",
        help="after the sync, add genomes/<genome_id>.fasta symlinks for .fna/.fa or sub-folder genomes",
    )
    parser.add_argument(
        "--aws-profile", dest="aws_profile", default=None, metavar="NAME",
        help="named AWS CLI profile for an s3:// source; set as AWS_PROFILE for the 'aws s3 sync' subprocess only",
    )


def _add_ingest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-ns", type=int, default=50, help="min non-susceptible rows per pair")
    parser.add_argument("--min-s", type=int, default=50, help="min susceptible rows per pair")
    parser.add_argument("--min-levels", type=int, default=4, help="min distinct MIC levels per pair")


def _add_lineage_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--threshold", type=float, default=0.005, help="single-linkage Mash distance cut")
    parser.add_argument("--lineage-backend", dest="lineage_backend", default="mash_single_linkage", choices=["mash_single_linkage", "poppunk"])
    parser.add_argument("--sketch-size", type=int, default=1000)


def _add_split_args(parser: argparse.ArgumentParser, *, with_force: bool = True) -> None:
    parser.add_argument("--split-seed", dest="split_seed", type=int, default=7, help="seed for test clusters and folds")
    parser.add_argument("--test-range", type=float, nargs=2, default=(0.15, 0.20), metavar=("LO", "HI"))
    parser.add_argument("--n-folds", type=int, default=5)
    if with_force:
        parser.add_argument("--force", action="store_true", help="overwrite the frozen splits.parquet (invalidates every result)")


def _add_unitig_args(parser: argparse.ArgumentParser, *, with_species: bool = True) -> None:
    parser.add_argument(
        "--unitig-backend", dest="unitig_backend", default="kmer", choices=["kmer", "unitig-caller", "auto"],
        help="kmer: pure Python (small sets only); unitig-caller: needs the tool on PATH (use at scale); "
             "auto: unitig-caller when installed, else kmer",
    )
    parser.add_argument(
        "--unitig-max-kmer-genomes", dest="unitig_max_kmer_genomes", type=int, default=1000, metavar="N",
        help="the kmer backend refuses species with more than N training genomes (default: 1000)",
    )
    parser.add_argument(
        "--unitig-threads", dest="unitig_threads", type=int, default=None, metavar="N",
        help="worker processes for per-genome k-mer work and unitig-caller --threads (default: all cores)",
    )
    if with_species:
        parser.add_argument("--species", nargs="*", default=None, help="species keys to build (default: all in splits)")


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", nargs="*", default=None, help="model ids (default: b1_lookup b2_xgb_steps aft_known aft_known_unitig)")
    parser.add_argument("--top-k", type=int, default=2000, help="unitig columns kept per fold")
    parser.add_argument("--min-count", type=int, default=5, help="known-AMR rare-feature filter (training genomes)")
    parser.add_argument("--nthread", type=int, default=4, help="xgboost threads")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="train the drugs of a species in this many worker processes (each uses --nthread "
             "xgboost threads); results are identical to --workers 1, only wall time changes",
    )
    parser.add_argument(
        "--exact-weight", dest="exact_weight", type=float, default=2.0,
        help="sample weight of exact-MIC rows relative to censored / S/I/R-only rows in the AFT models "
             "(default 2.0; 1.0 = unweighted)",
    )
    parser.add_argument(
        "--band", choices=("asym_tuned", "symmetric"), default="asym_tuned",
        help="uncertainty band: asym_tuned (default; asymmetric cross-conformal band whose upper level is "
             "tuned inside the training folds for call-level VME <= 1.5%%, active calls withheld when no "
             "level certifies it) or symmetric (the original +-q band at 90%%)",
    )
    parser.add_argument(
        "--no-model-select", dest="no_model_select", action="store_true",
        help="ship the AFT model as is instead of aft_b2_select (the per-pair choice of AFT, B2 or their average "
             "made inside the training folds; default on)",
    )
    parser.add_argument("--train-seed", dest="train_seed", type=int, default=7, help="seed for in-fold holdouts and xgboost")
    parser.add_argument("--max-rounds", type=int, default=400)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    parser.add_argument("--no-lolo", action="store_true", help="skip leave-one-lineage-out runs")
    parser.add_argument(
        "--cv-only", dest="cv_only", action="store_true",
        help="fit CV folds (OOF split='cv'), conformal and the final bundle on all train rows, but never "
             "predict test or LOLO rows and never write the test ledger",
    )
    parser.add_argument("--ablation", action="store_true", help="also run aft_unitig_only")
    parser.add_argument(
        "--amrfinder-db", dest="amrfinder_db", default=None,
        help="AMRFinderPlus database dir (fam.tsv, AMRProt.fa) giving every gene_ column its Subclass, so "
             "training-time calls apply drugs.yaml strong_subclasses like the prediction pipeline "
             "(default: next to amrfinder on PATH; none -> prefix rule only, with a warning)",
    )
    parser.add_argument(
        "--species", dest="train_species", nargs="+", default=None, metavar="KEY",
        help="train only the kept pairs of these species (a shard; models/manifest.json is merged, not replaced)",
    )
    parser.add_argument(
        "--drugs", dest="train_drugs", nargs="+", default=None, metavar="DRUG",
        help="train only the kept pairs of these drugs (combine with --species; merged like --species)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genome2mic",
        description="Genome -> MIC prediction pipeline. Predictions of in-vitro susceptibility, not prescribing advice.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="write the seeded synthetic raw layer under --root")
    _add_common(p); _add_synth_args(p); p.set_defaults(func=cmd_synth)

    p = sub.add_parser("fetch", help="sync ast_*.csv + genomes/ from a bucket or directory into <root>/data/raw")
    _add_common(p); _add_fetch_args(p, required=True)
    p.add_argument("--dry-run", action="store_true", help="print the sync command and run nothing")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser(
        "import-release",
        help="verify (SHA256SUMS) and copy a frozen data release into <root>/data/processed (+ qc.parquet)",
    )
    _add_common(p)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--release-dir", default=None, help="local release directory (read only), e.g. a develop checkout's data/processed")
    source.add_argument("--s3-release", default=None, metavar="NAME", help="download s3://g2m-data-v1/releases/NAME/ with aws s3 sync first")
    p.add_argument("--aws-profile", dest="aws_profile", default=None, metavar="NAME", help="AWS CLI profile for --s3-release")
    p.set_defaults(func=cmd_import_release)

    p = sub.add_parser(
        "release-feature-spec",
        help="bundle trained on an imported NCBI release: write models/<SP>/feature_spec.json (release feature rules)",
    )
    _add_common(p)
    p.add_argument("--amrfinder-db", default=None, help="AMRFinderPlus database dir (default: next to amrfinder on PATH)")
    p.set_defaults(func=cmd_release_feature_spec)

    p = sub.add_parser("ingest", help="raw AST -> labels.parquet, label_counts.csv, pairs_kept.csv")
    _add_common(p); _add_ingest_args(p); p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("qc", help="assembly stats + species ID -> qc.parquet")
    _add_common(p); p.set_defaults(func=cmd_qc)

    p = sub.add_parser("known-amr", help="AMRFinderPlus output -> known_amr.parquet")
    _add_common(p); p.set_defaults(func=cmd_known_amr)

    p = sub.add_parser("lineages", help="sketches + single-linkage clusters -> lineages.parquet")
    _add_common(p); _add_lineage_args(p); p.set_defaults(func=cmd_lineages)

    p = sub.add_parser("splits", help="lineage-grouped train/test + folds -> splits.parquet (frozen)")
    _add_common(p); _add_split_args(p); p.set_defaults(func=cmd_splits)

    p = sub.add_parser("unitigs", help="k-mer/unitig patterns built on training genomes only")
    _add_common(p); _add_unitig_args(p); p.set_defaults(func=cmd_unitigs)

    p = sub.add_parser("train", help="CV + final fits -> results/preds_*.parquet and models/")
    _add_common(p); _add_train_args(p); p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="preds -> metrics.parquet (VME first) + metrics_by_distance.parquet")
    _add_common(p); p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("report", help="metrics + figures -> results/report.md")
    _add_common(p); p.add_argument("--no-figures", action="store_true"); p.set_defaults(func=cmd_report)

    p = sub.add_parser(
        "compare-oof",
        help="our split='cv' preds vs another model's OOF preds on identical (genome_id, drug) rows and breakpoints",
    )
    _add_common(p)
    p.add_argument("--develop-preds", required=True, help="OOF preds parquet of the reference model (e.g. develop's multitask)")
    p.add_argument("--breakpoints", default=None, help="breakpoint CSV to use instead of the call standard (species,drug,s_breakpoint,r_breakpoint)")
    p.add_argument("--out", required=True, help="output directory for oof_compare.csv / oof_compare.md")
    p.add_argument("--amrfinder-db", dest="amrfinder_db", default=None,
                   help="AMRFinderPlus database dir: recomputed calls apply strong_subclasses like training")
    p.set_defaults(func=cmd_compare_oof)

    p = sub.add_parser("predict", help="one FASTA -> report JSON on stdout")
    _add_common(p)
    p.add_argument("--fasta", required=True)
    p.add_argument("--sample-id", default=None, help="default: FASTA stem")
    p.add_argument("--amrfinder-tsv", default=None, help="precomputed AMRFinderPlus TSV for this genome")
    p.add_argument("--models-dir", default=None, help="default: <root>/models")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("run-all", help="synth (or fetch, with --source-uri) + every pipeline stage in order")
    _add_common(p); _add_synth_args(p); _add_ingest_args(p); _add_lineage_args(p)
    _add_split_args(p, with_force=False); _add_unitig_args(p, with_species=False); _add_train_args(p)
    _add_fetch_args(p, required=False)
    p.add_argument("--no-synth", action="store_true", help="skip synthetic data generation (use existing raw data)")
    p.add_argument("--force-splits", action="store_true", help="overwrite an existing splits.parquet")
    p.add_argument("--no-figures", action="store_true")
    p.set_defaults(func=cmd_run_all)
    return parser


def _configure_logging(args: argparse.Namespace) -> None:
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING if getattr(args, "quiet", False) else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
