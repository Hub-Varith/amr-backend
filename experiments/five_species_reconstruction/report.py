import json
from pathlib import Path
root=Path(__file__).resolve().parent
samples=json.loads((root/'samples.json').read_text())
lines=['# Five-species assembly batch','','Research assembly baseline. No learned model or MIC validation. These are research inputs for future predictions of in-vitro susceptibility, not prescribing advice.','','| Species | Reads used | Status | Contigs | N50 |','|---|---:|---|---:|---:|']
for species in samples:
    for level,fraction in [('00','25%'),('01','50%'),('02','100%')]:
        s=root/f'status/{species}_{level}.json';m=root/f'results/{species}/{level}/stats.json'
        state=json.loads(s.read_text())['status'] if s.exists() else 'queued / not started'
        metrics=json.loads(m.read_text()) if m.exists() else {}
        lines.append(f"| {species} | {fraction} | {state} | {metrics.get('contigs','—')} | {metrics.get('n50','—')} |")
lines+=['','The four new species do not yet have validated comparison references. Contiguity is not accuracy. One run per species and one subsampling seed do not establish general performance. Logs contain download/assembly errors; missing output is not success.','']
(root/'STATUS.md').write_text('\n'.join(lines))
print('\n'.join(lines))
