"""Create nested subsets of single-end four-line FASTQ reads; no MIC prediction."""
import argparse
import gzip
import json
import math
import random
from contextlib import ExitStack
from pathlib import Path


def records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        while True:
            header = handle.readline()
            if not header:
                return
            sequence, plus, quality = (handle.readline() for _ in range(3))
            seq, qual = sequence.rstrip("\r\n"), quality.rstrip("\r\n")
            if (not header.startswith("@") or not plus.startswith("+")
                    or not seq or not quality or len(seq) != len(qual)):
                raise ValueError("Malformed FASTQ: expected four-line records with equal sequence/quality lengths")
            yield header + sequence + plus + quality, len(seq)


def subsample(source, output, fractions, seed=42, mode="random"):
    source, output = Path(source), Path(output)
    fractions = sorted(set(fractions))
    if not fractions or any(not math.isfinite(f) or not 0 < f <= 1 for f in fractions):
        raise ValueError("Fractions must be finite numbers in (0, 1]")
    if mode not in {"random", "prefix"}:
        raise ValueError("Mode must be random or prefix")
    # Validate the entire input before creating outputs; count for prefix mode.
    total_reads = total_bases = 0
    for _, bases in records(source):
        total_reads += 1
        total_bases += bases
    if not total_reads:
        raise ValueError("Input FASTQ is empty")
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    entries = [dict(fraction=f, file=f"subset_{i:02d}.fastq", reads=0, bases=0)
               for i, f in enumerate(fractions)]
    with ExitStack() as stack:
        handles = [stack.enter_context((output / e["file"]).open("w")) for e in entries]
        for index, (record, bases) in enumerate(records(source)):
            draw = rng.random()
            for entry, handle in zip(entries, handles):
                keep = (draw < entry["fraction"] if mode == "random"
                        else index < math.ceil(entry["fraction"] * total_reads))
                if keep:
                    handle.write(record)
                    entry["reads"] += 1
                    entry["bases"] += bases
    manifest = dict(source=str(source.resolve()), mode=mode, seed=seed,
                    total_reads=total_reads, total_bases=total_bases, subsets=entries)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, help="New directory; existing directories are refused")
    parser.add_argument("--fractions", type=float, nargs="+", default=[.01, .05, .1, .25, .5, 1.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["random", "prefix"], default="random")
    args = parser.parse_args()
    manifest = subsample(args.source, args.output, args.fractions, args.seed, args.mode)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
