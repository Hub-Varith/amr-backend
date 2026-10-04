"""Hackathon end-to-end demo: unseen genomes -> prediction CLI -> comparison with the lab.

These are predictions of in-vitro susceptibility, not prescribing advice.

Subcommands (run from the repo root with the project venv)::

    python scripts/hackathon_demo.py select  --root runs/hackathon5        # demo/genomes.csv
    python scripts/hackathon_demo.py fetch   --root runs/hackathon5        # demo/genomes/<id>.fasta (BV-BRC API)
    python scripts/hackathon_demo.py predict --root runs/hackathon5 --jobs 4  # demo/<id>.json via the CLI
    python scripts/hackathon_demo.py compare --root runs/hackathon5        # demo/demo_table.{csv,md}

Selection rule (never the test split, never a training genome): per species, BV-BRC genome
ids that have lab results in ``labels.parquet`` for a kept species x drug pair but are absent
from ``splits.parquet``. They were never seen by training (training only uses split rows) and
are not test rows. Per species the script takes, from candidates sorted by number of
labelled drugs (desc) then genome_id: the first with >= 2 lab-R drugs, the first with no
lab-R drug, and the first remaining genome with >= 1 lab-R drug (else the next one).
Lab S/I/R is re-derived from the MIC interval under the call standard (CLSI 2024) when the
interval allows it, else the lab's reported S/I/R.

``fetch`` downloads contigs from ``https://www.bv-brc.org/api/genome_sequence/`` with
``Accept: application/dna+fasta``. ``predict`` runs ``python -m genome2mic predict`` (AMRFinderPlus
must be on PATH; the script prepends ``$HOME/micromamba/envs/amrfinder/bin`` when present).

This is a tiny unseen-genome check of the plumbing, not a validation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from genome2mic.config import load_config  # noqa: E402
from genome2mic.models.train import rederive_lab_sir  # noqa: E402

BVBRC_URL = "https://www.bv-brc.org/api/genome_sequence/?eq(genome_id,{gid})&limit(25000)"
USER_AGENT = "genome2mic-hackathon-demo/0.1 (research; contact via repository)"
AMRFINDER_BIN = Path.home() / "micromamba" / "envs" / "amrfinder" / "bin"
SPECIES = ("KPNEU", "ECOLI", "SAUR", "PAER", "ABAU")


def _lab_table(root: Path) -> pd.DataFrame:
    processed = root / "data" / "processed"
    labels = pd.read_parquet(processed / "labels.parquet")
    pairs = pd.read_csv(processed / "pairs_kept.csv")[["species", "drug"]]
    labels = labels.merge(pairs, on=["species", "drug"])
    config = load_config(REPO / "configs")
    sir = []
    for row in labels.itertuples(index=False):
        bp = config.call_breakpoint(row.species, row.drug)
        derived = rederive_lab_sir(row.mic_lower, row.mic_upper, bp)
        sir.append(derived if derived is not None else (row.sir if isinstance(row.sir, str) else None))
    labels["lab_sir"] = sir
    return labels


def cmd_select(args: argparse.Namespace) -> None:
    root = Path(args.root)
    labels = _lab_table(root)
    splits = pd.read_parquet(root / "data" / "processed" / "splits.parquet")
    excluded = [g for g in (args.exclude or "").split(",") if g]
    unseen = labels[~labels["genome_id"].isin(splits["genome_id"]) & ~labels["genome_id"].str.startswith("NCBI_")]
    unseen = unseen[~unseen["genome_id"].isin(excluded)]
    chosen = []
    for species in SPECIES:
        sp = unseen[unseen["species"] == species]
        per = sp.groupby("genome_id").agg(n_drugs=("drug", "nunique"), n_r=("lab_sir", lambda s: int((s == "R").sum())))
        per = per.reset_index().sort_values(["n_drugs", "genome_id"], ascending=[False, True])
        picks: list[str] = []
        for rule in (lambda r: r.n_r >= 2, lambda r: r.n_r == 0, lambda r: r.n_r >= 1, lambda r: True):
            for r in per.itertuples(index=False):
                if r.genome_id not in picks and rule(r):
                    picks.append(r.genome_id)
                    break
            if len(picks) == 3:
                break
        for gid in picks[:3]:
            row = per[per["genome_id"] == gid].iloc[0]
            chosen.append({"species": species, "genome_id": gid, "n_drugs": int(row.n_drugs), "n_lab_r": int(row.n_r)})
        print(f"{species}: {len(sp['genome_id'].unique())} unseen BV-BRC candidates -> {picks[:3]}")
    out = root / "demo"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(chosen).to_csv(out / "genomes.csv", index=False)
    print(f"wrote {out / 'genomes.csv'}")
    if excluded:
        pd.DataFrame({"genome_id": excluded}).to_csv(out / "genomes_excluded.csv", index=False)
        print(f"wrote {out / 'genomes_excluded.csv'} (picked again without these ids)")


def _download(gid: str, dest: Path) -> str:
    if dest.is_file() and dest.stat().st_size > 0:
        return f"{gid}: cached"
    request = urllib.request.Request(BVBRC_URL.format(gid=gid), headers={"Accept": "application/dna+fasta", "User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read()
            if not body.startswith(b">"):
                raise ValueError(f"not FASTA: {body[:80]!r}")
            dest.write_bytes(body)
            return f"{gid}: {body.count(b'>')} contigs, {len(body)} bytes"
        except Exception as error:  # network hiccups: retry, then report
            if attempt == 2:
                return f"{gid}: FAILED ({error})"
            time.sleep(3)
    return f"{gid}: FAILED"


def cmd_fetch(args: argparse.Namespace) -> None:
    root = Path(args.root)
    table = pd.read_csv(Path(args.genomes) if args.genomes else root / "demo" / "genomes.csv", dtype=str)
    gdir = Path(args.out_dir) if args.out_dir else root / "demo" / "genomes"
    gdir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for message in pool.map(lambda g: _download(g, gdir / f"{g}.fasta"), table["genome_id"]):
            print(message)


def _env() -> dict[str, str]:
    env = dict(os.environ)
    if AMRFINDER_BIN.is_dir():
        env["PATH"] = f"{AMRFINDER_BIN}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{REPO / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return env


def _predict_one(root: Path, gid: str) -> str:
    fasta = root / "demo" / "genomes" / f"{gid}.fasta"
    out = root / "demo" / f"{gid}.json"
    log = root / "demo" / f"{gid}.log"
    command = [sys.executable, "-m", "genome2mic", "predict", "--root", str(root), "--fasta", str(fasta), "--sample-id", gid]
    started = time.time()
    with out.open("w") as stdout, log.open("w") as stderr:
        code = subprocess.run(command, stdout=stdout, stderr=stderr, env=_env(), cwd=REPO).returncode
    return f"{gid}: exit {code} in {time.time() - started:.0f}s -> {out}"


def cmd_predict(args: argparse.Namespace) -> None:
    root = Path(args.root)
    table = pd.read_csv(root / "demo" / "genomes.csv", dtype=str)
    with ThreadPoolExecutor(max_workers=int(args.jobs)) as pool:
        for message in pool.map(lambda g: _predict_one(root, g), table["genome_id"]):
            print(message)


def _fmt_mic(lo: float, hi: float) -> str:
    if lo <= 0:
        return f"<={hi:g}"
    if not np.isfinite(hi):
        return f">{lo:g}"
    return f"{hi:g}" if hi == 2 * lo else f"({lo:g},{hi:g}]"


def cmd_compare(args: argparse.Namespace) -> None:
    root = Path(args.root)
    table = pd.read_csv(root / "demo" / "genomes.csv", dtype=str)
    labels = _lab_table(root)
    rows = []
    summary = []
    for record in table.itertuples(index=False):
        report_path = root / "demo" / f"{record.genome_id}.json"
        try:
            report = json.loads(report_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            summary.append({"genome_id": record.genome_id, "species_lab": record.species, "species_pred": None, "note": "no report"})
            continue
        preds = {p["drug"]: p for p in report["predictions"]}
        lab = labels[labels["genome_id"] == record.genome_id]
        summary.append({
            "genome_id": record.genome_id,
            "species_lab": record.species,
            "species_pred": report["species"],
            "qc_pass": report["qc_pass"],
            "n_predicted": len(report["predictions"]),
            "ranked_active": ", ".join(r["drug"] if isinstance(r, dict) else str(r) for r in report["ranked_active"][:5]),
        })
        for lab_row in lab.itertuples(index=False):
            pred = preds.get(lab_row.drug)
            call = pred["call"] if pred else None
            lab_sir = lab_row.lab_sir
            outcome = None
            if call is not None and lab_sir in ("S", "R", "I"):
                if lab_sir == "R" and call == "likely_active":
                    outcome = "VME"
                elif lab_sir == "S" and call == "likely_inactive":
                    outcome = "ME"
                elif call == "uncertain":
                    outcome = "uncertain"
                elif (lab_sir == "R" and call == "likely_inactive") or (lab_sir == "S" and call == "likely_active"):
                    outcome = "agree"
                else:
                    outcome = "lab I"
            ea = None
            if pred and pred.get("pred_mic") is not None and lab_row.mic_lower > 0 and np.isfinite(lab_row.mic_upper) \
                    and lab_row.mic_upper == 2 * lab_row.mic_lower:
                ea = bool(abs(np.log2(pred["pred_mic"]) - np.log2(lab_row.mic_upper)) <= 1 + 1e-9)
            rows.append({
                "genome_id": record.genome_id,
                "species": record.species,
                "drug": lab_row.drug,
                "lab_mic": _fmt_mic(lab_row.mic_lower, lab_row.mic_upper),
                "lab_sir": lab_sir,
                "pred_mic": pred.get("pred_mic") if pred else None,
                "band": f"{pred['band_low']:g}-{pred['band_high']:g}" if pred and pred.get("band_low") is not None else None,
                "call": call,
                "override": pred.get("override") if pred else None,
                "outcome": outcome,
                "within_1_step": ea,
            })
    detail = pd.DataFrame(rows)
    out = root / "demo"
    detail.to_csv(out / "demo_table.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "demo_genomes_summary.csv", index=False)
    per_genome = (
        detail.groupby(["species", "genome_id"], sort=False)
        .agg(
            n_lab=("drug", "size"),
            vme=("outcome", lambda s: int((s == "VME").sum())),
            me=("outcome", lambda s: int((s == "ME").sum())),
            agree=("outcome", lambda s: int((s == "agree").sum())),
            uncertain=("outcome", lambda s: int((s == "uncertain").sum())),
            no_call=("call", lambda s: int(s.isna().sum())),
            ea=("within_1_step", lambda s: f"{int(s.fillna(False).astype(bool).sum())}/{int(s.notna().sum())}"),
        )
        .reset_index()
    )
    per_genome = per_genome.merge(pd.DataFrame(summary)[["genome_id", "species_pred", "ranked_active"]], on="genome_id", how="left")
    per_genome = per_genome[["species", "genome_id", "species_pred", "n_lab", "vme", "me", "agree", "uncertain", "no_call", "ea", "ranked_active"]]
    per_genome.to_csv(out / "demo_per_genome.csv", index=False)
    print(per_genome.to_markdown(index=False, disable_numparse=True))
    excluded_csv = out / "genomes_excluded.csv"
    if excluded_csv.is_file():
        rejected = []
        for gid in pd.read_csv(excluded_csv, dtype=str)["genome_id"]:
            log = out / f"{gid}.log"
            line = next((l for l in log.read_text().splitlines() if "reference distance" in l), "") if log.is_file() else ""
            report = json.loads((out / f"{gid}.json").read_text()) if (out / f"{gid}.json").is_file() else {}
            species_lab = labels.loc[labels["genome_id"] == gid, "species"].iloc[0]
            rejected.append({"genome_id": gid, "species_lab": species_lab, "species_pred": report.get("species"),
                             "qc_pass": report.get("qc_pass"), "n_predictions": len(report.get("predictions", [])),
                             "qc_line": line.split("INFO genome2mic.predict.pipeline: ")[-1]})
        rejected = pd.DataFrame(rejected)
        rejected.to_csv(out / "demo_qc_rejected.csv", index=False)
        print(rejected.to_markdown(index=False, disable_numparse=True))
    totals = detail["outcome"].value_counts().to_dict()
    print("totals:", totals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("select", cmd_select), ("fetch", cmd_fetch), ("predict", cmd_predict), ("compare", cmd_compare)):
        p = sub.add_parser(name)
        p.add_argument("--root", default="runs/hackathon5")
        p.set_defaults(func=func)
        if name == "select":
            p.add_argument("--exclude", default=None,
                           help="comma-separated genome ids to skip (e.g. genomes the pipeline's species QC rejected)")
        if name == "fetch":
            p.add_argument("--genomes", default=None, help="CSV with a genome_id column (default: <root>/demo/genomes.csv)")
            p.add_argument("--out-dir", default=None, help="default: <root>/demo/genomes")
        if name == "predict":
            p.add_argument("--jobs", default=3)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
