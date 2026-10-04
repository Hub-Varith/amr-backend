"""Predict one genome from the command line (the same pipeline the API runs).

Example (Linux, with mash and amrfinder on PATH):
    python -m genome2mic.predict.run_predict --fasta sample.fasta --sample-id BC-0142 --out report.json

These are predictions of in-vitro susceptibility, not prescribing advice.
"""

import argparse
import json
import logging
from pathlib import Path

from genome2mic.api.schemas.prediction_report import PredictionReport
from genome2mic.predict.pipeline import PredictionPipeline


def print_report(report: PredictionReport) -> None:
    print(f"Sample: {report.sample_id}   Species: {report.species.value if report.species else 'not covered'}   "
          f"QC: {'pass' if report.qc_pass else 'FAIL'}   In range: {report.in_range}   "
          f"Nearest training genome: {report.nearest_training_distance if report.nearest_training_distance is not None else 'not checked'}")
    if report.predictions:
        print(f"\n{'Drug':32}{'MIC (90% band) mg/L':26}{'S bp':>7}  {'p_active':>8}  Call")
        for row in report.predictions:
            mic = "-" if row.pred_mic is None else f"{row.pred_mic:g} ({row.band_low:g}-{row.band_high:g})"
            s_bp = "-" if row.s_breakpoint is None else f"<={row.s_breakpoint:g}"
            p_active = "-" if row.p_active is None else f"{row.p_active:.2f}"
            note = f" [{row.override.value}]" if row.override else ""
            reasons = f"  ({', '.join(row.reasons)})" if row.reasons else ""
            print(f"{row.drug:32}{mic:26}{s_bp:>7}  {p_active:>8}  {row.call.value}{note}{reasons}")
    print(f"\nRanked likely-active: {', '.join(report.ranked_active) or 'none'}")
    print(f"Model {report.model_version} (run {report.run_id})")
    print(f"\n{report.disclaimer}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fasta", type=Path, required=True, help="Assembled genome (contigs), .fasta/.fa/.fna")
    parser.add_argument("--sample-id", default=None, help="Default: the file name without extension")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--model-run", default="all5_run1")
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--references-sketch", type=Path, default=Path("data/references/references.msh"))
    parser.add_argument("--amrfinder-db", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--keep-work", action="store_true", help="Keep qc.json, amrfinder.tsv, known_amr.json, ...")
    parser.add_argument("--out", type=Path, default=None, help="Write the report JSON here")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    pipeline = PredictionPipeline(
        arguments.models_dir, arguments.configs_dir, model_run=arguments.model_run,
        references_sketch=arguments.references_sketch, amrfinder_db=arguments.amrfinder_db,
        threads=arguments.threads, keep_work_files=arguments.keep_work,
    )
    pipeline.load()
    report = PredictionReport.model_validate(pipeline.run(arguments.fasta, arguments.sample_id or arguments.fasta.stem))
    if arguments.out:
        arguments.out.write_text(json.dumps(report.model_dump(mode="json"), indent=2))
    print_report(report)


if __name__ == "__main__":
    main()
