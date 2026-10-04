"""Runs ``workflow/Snakefile`` against fake ``mash`` / ``amrfinder`` executables (scale #24, #30).

The real tools are not installed here, so tiny shell stand-ins record their arguments
and emit output in the tools' formats. This checks the command lines the Snakefile
builds -- the part that had never run:

* species references are sketched ONCE into one ``.msh`` (``mash sketch -o``) and each
  genome is compared with ``mash dist <refs.msh> <genome>``, so every reference appears
  in ``mash.tsv`` (the old command made the first reference FASTA the only reference);
* ``qc`` reads that table and picks the right species;
* ``amrfinder`` gets ``-O <organism>`` from ``genome_metadata.csv`` and ``--threads``;
* ``--config synthetic=false`` (a string for snakemake) really means "not synthetic".
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from genome2mic import qc

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAKEFILE = REPO_ROOT / "workflow" / "Snakefile"
SNAKEMAKE = shutil.which("snakemake") or str(Path(sys.executable).with_name("snakemake"))

FAKE_MASH = """#!/bin/sh
# fake mash. sketch -o PREFIX FILES...: PREFIX.msh lists FILES. dist REF.msh QUERY: one row per reference.
echo "mash $*" >> "$FAKE_LOG"
if [ "$1" = "sketch" ]; then
  shift; [ "$1" = "-o" ] || exit 2; prefix="$2"; shift 2
  : > "$prefix.msh"; for f in "$@"; do echo "$f" >> "$prefix.msh"; done
elif [ "$1" = "dist" ]; then
  case "$2" in *.msh) ;; *) echo "reference must be the .msh" >&2; exit 3;; esac
  while read -r r; do
    case "$r" in *KPNEU*) d=0.004;; *) d=0.31;; esac
    printf "%s\\t%s\\t%s\\t0\\t900/1000\\n" "$r" "$3" "$d"
  done < "$2"
fi
"""

FAKE_AMRFINDER = """#!/bin/sh
echo "amrfinder $*" >> "$FAKE_LOG"
while [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; shift; done
printf "Name\\tProtein identifier\\n" > "$out"
"""


@pytest.mark.skipif(not Path(SNAKEMAKE).exists(), reason="snakemake not installed")
def test_snakefile_sketches_references_once_and_compares_every_reference(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("mash", FAKE_MASH), ("amrfinder", FAKE_AMRFINDER)):
        exe = bin_dir / name
        exe.write_text(body)
        exe.chmod(0o755)
    root = tmp_path / "root"
    genomes = root / "data" / "raw" / "genomes"
    refs = root / "data" / "raw" / "references"
    genomes.mkdir(parents=True)
    refs.mkdir(parents=True)
    for gid in ("g1", "g2"):
        (genomes / f"{gid}.fasta").write_text(">c1\nACGTACGTAC\n")
    for key in ("ABAU", "ECOLI", "KPNEU"):  # ABAU sorts first: the old command used it as the only reference
        (refs / f"{key}.fasta").write_text(">r\nACGT\n")
    (root / "data" / "raw" / "genome_metadata.csv").write_text("genome_id,biosample,species\ng1,S1,KPNEU\ng2,S2,ECOLI\n")
    log = tmp_path / "calls.log"
    interim = root / "data" / "interim"
    targets = [str(interim / g / f) for g in ("g1", "g2") for f in ("mash.tsv", "amrfinder.tsv")]
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}", "FAKE_LOG": str(log)}
    proc = subprocess.run(
        [SNAKEMAKE, *targets, "-s", str(SNAKEFILE), "--directory", str(tmp_path), "--cores", "4",
         "--config", f"root={root}", "synthetic=false", f"configs_dir={REPO_ROOT / 'configs'}", "amrfinder_threads=2"],
        env=env, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]

    calls = log.read_text().splitlines()
    sketch_calls = [c for c in calls if c.startswith("mash sketch")]
    assert len(sketch_calls) == 1, calls  # references sketched once, not once per genome
    assert all(f"{key}.fasta" in sketch_calls[0] for key in ("ABAU", "ECOLI", "KPNEU"))
    dist_calls = [c for c in calls if c.startswith("mash dist")]
    assert len(dist_calls) == 2 and all(c.split()[2].endswith("references.msh") for c in dist_calls)

    table = qc.read_mash_tsv(interim / "g1" / "mash.tsv")
    assert sorted(Path(ref).stem for ref, _ in table) == ["ABAU", "ECOLI", "KPNEU"]  # every reference compared
    assert qc.species_from_mash_tsv(interim / "g1" / "mash.tsv", ["ABAU", "ECOLI", "KPNEU"]) == ("KPNEU", 0.004)

    amr = sorted(c for c in calls if c.startswith("amrfinder"))
    assert len(amr) == 2
    assert "-O Klebsiella_pneumoniae" in " ".join(amr) and "-O Escherichia" in " ".join(amr)
    assert all("--threads 2" in c for c in amr)
