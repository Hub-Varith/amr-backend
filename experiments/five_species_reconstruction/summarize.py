"""Descriptive reference alignment metrics, not AMR or clinical validation."""
import json,sys
from pathlib import Path

def fasta(path):
    out={}; name=None
    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            name=line[1:].split()[0]; out[name]=''
        elif line.strip():
            if name is None: raise ValueError('Invalid FASTA')
            out[name]+=line.strip()
    return out

def union_length(intervals):
    total=0; end=0
    for a,b in sorted(intervals):
        total+=max(0,b-max(a,end)); end=max(end,b)
    return total

def summarize(ref_path,asm_path,paf_path):
    refs=fasta(ref_path); asm=fasta(asm_path)
    targets={k:[] for k in refs}; queries={k:[] for k in asm}
    matches=columns=0
    for line in Path(paf_path).read_text().splitlines():
        p=line.split('\t')
        if 'tp:A:P' not in p[12:]: continue
        targets[p[5]].append((int(p[7]),int(p[8])))
        queries[p[0]].append((int(p[2]),int(p[3])))
        matches+=int(p[9]); columns+=int(p[10])
    ref_total=sum(map(len,refs.values())); asm_total=sum(map(len,asm.values()))
    return dict(contigs=len(asm),assembly_bases=asm_total,reference_bases=ref_total,
        reference_covered_percent=100*sum(map(union_length,targets.values()))/ref_total,
        assembly_aligned_percent=100*sum(map(union_length,queries.values()))/asm_total if asm_total else 0,
        aligned_identity_percent=100*matches/columns if columns else None,
        per_reference_covered_percent={k:100*union_length(targets[k])/len(v) for k,v in refs.items()},
        note='Primary alignment-span coverage and alignment-weighted identity; not a structural-error or AMR assessment.')

if __name__=='__main__':
    print(json.dumps(summarize(*sys.argv[1:]),indent=2))
