import json,sys
from pathlib import Path
from summarize import fasta
asm=Path(sys.argv[1]); dest=Path(sys.argv[2]); seq=fasta(asm)
sizes=sorted(map(len,seq.values()),reverse=True);total=sum(sizes);running=0;n50=0
for size in sizes:
    running+=size
    if running>=total/2:n50=size;break
if not sizes: raise ValueError('Empty assembly')
dest.write_text(json.dumps(dict(contigs=len(sizes),assembly_bases=total,largest_contig=max(sizes),n50=n50,accuracy_evaluated=False),indent=2)+'\n')
