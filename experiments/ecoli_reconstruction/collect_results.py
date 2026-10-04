"""Collect completed pilot metrics without silently excluding failed runs."""
import json
from pathlib import Path
from summarize import fasta

root=Path(__file__).resolve().parent
manifest=json.loads((root/'subsets/manifest.json').read_text())
ref_length=sum(map(len,fasta(root/'reference.fasta').values()))
rows=[]
for i,subset in enumerate(manifest['subsets']):
    path=root/f'results/{i:02d}/metrics.json'
    rows.append(dict(fraction=subset['fraction'],reads=subset['reads'],bases=subset['bases'],
        average_depth=subset['bases']/ref_length,status='completed' if path.exists() else 'missing_or_failed',
        metrics=json.loads(path.read_text()) if path.exists() else None))
(root/'benchmark_results.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['# Pilot results','', 'Flye 2.9.6; one E. coli mock-community sample, one subsampling seed. No model training or MIC evaluation.', '', '| Reads used | Average depth | Contigs | Reference aligned breadth | Aligned sequence identity |', '|---|---:|---:|---:|---:|']
for r in rows:
    m=r['metrics']
    if m is None:
        lines.append(f"| {r['fraction']:.0%} | {r['average_depth']:.2f}x | missing/failed | — | — |")
    else:
        identity='unavailable' if m['aligned_identity_percent'] is None else f"{m['aligned_identity_percent']:.4f}%"
        lines.append(f"| {r['fraction']:.0%} | {r['average_depth']:.2f}x | {m['contigs']} | {m['reference_covered_percent']:.4f}% | {identity} |")
lines+=['','Alignment breadth and identity are different measures. These results do not establish complete correctness, resistance-marker preservation, MIC accuracy, or a validated minimum sequencing depth. See README.md for provenance and limitations.','']
(root/'RESULTS.md').write_text('\n'.join(lines))
print('\n'.join(lines))
