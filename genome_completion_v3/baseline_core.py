"""Inference sees observed sequence and training references ONLY.

No test truth, target IDs, target coordinates, labels, or MIC values are accepted.
This module is deliberately separate from the benchmark's truth preparation.
"""
import gzip, hashlib, json, math, os, re, subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np

MM2 = os.environ.get('COMPLETION_MINIMAP2', '/tmp/amr-flye-2.9.6/bin/flye-minimap2')
FEATURES = ['probe_fraction','mean_identity','min_identity','mean_probe_coverage',
            'left_identity','right_identity','left_coverage','right_coverage',
            'strand_agreement','observed_span_ratio','predicted_length_ratio',
            'log_observed_length','predicted_known_fraction','endpoint_clip_fraction']
CS = re.compile(r':[0-9]+|=[A-Za-z]+|\*[A-Za-z][A-Za-z]|[+-][A-Za-z]+')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def reverse_complement(s): return s.translate(str.maketrans('ACGTN','TGCAN'))[::-1]

def fasta(path):
    op = gzip.open if str(path).endswith('.gz') else open
    result=[]; name=None; parts=[]
    with op(path,'rt') as f:
        for line in f:
            if line.startswith('>'):
                if name is not None:result.append((name,''.join(parts).upper()))
                name=line[1:].strip();parts=[]
            else: parts.append(line.strip())
    if name is not None:result.append((name,''.join(parts).upper()))
    if not result or any(not seq for _,seq in result):raise ValueError(f'Empty FASTA: {path}')
    return result

def write_fasta(path, records):
    with Path(path).open('w') as f:
        for name,seq in records:
            f.write('>'+name+'\n')
            for i in range(0,len(seq),80):f.write(seq[i:i+80]+'\n')

def circular_slice(seq,start,length):
    if not 0<=length<=len(seq):raise ValueError('Cannot traverse chromosome more than once')
    start%=len(seq)
    return (seq+seq)[start:start+length]

def expose(seq,seed,missing_fraction=.35):
    """Benchmark-only masking helper. Returns truth separately."""
    n=len(seq);cut=int(n*missing_fraction)
    if not 0<cut<n:raise ValueError('Invalid missing fraction')
    start=int(np.random.default_rng(seed).integers(n))
    rotated=circular_slice(seq,start,n)
    return rotated[:-cut],rotated[-cut:],start

def sketch(sequences,k=19):
    """Canonical 19-mer hash sampling, independent of sequence origin/orientation."""
    lut=np.full(256,4,dtype=np.uint8)
    for i,c in enumerate(b'ACGT'):lut[c]=i
    out=set()
    for seq in sequences:
        a=lut[np.frombuffer(seq.encode(),dtype=np.uint8)];n=len(a)-k+1
        if n<=0:continue
        f=np.zeros(n,dtype=np.uint64);r=np.zeros(n,dtype=np.uint64);valid=np.ones(n,dtype=bool)
        for j in range(k):
            v=a[j:j+n];valid&=v<4
            f|=v.astype(np.uint64) << (2*(k-j-1))
            r|=(3-v.astype(np.uint64)) << (2*j)
        h=np.minimum(f,r)
        with np.errstate(over='ignore'):
            h=(h^(h>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
            h=(h^(h>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
            h=h^(h>>np.uint64(31))
        out.update(map(int,h[valid&((h&np.uint64(2047))==0)]))
    return out

def run_map(reference,query,out,log,secondary=True):
    cmd=[MM2,'-x','asm20','-t','2','-c','--cs=short','-N','200','-p','0.1',
         '--secondary='+('yes' if secondary else 'no'),str(reference),str(query)]
    with open(out,'w') as o,open(log,'w') as e:
        subprocess.run(cmd,stdout=o,stderr=e,check=True,timeout=600)

def candidates(observed,panel,work,mode='chromosome',missing_fraction=.35):
    """panel rows: donor, group, complete, records. Never contains target truth."""
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    width=min(4000,max(200,len(observed)//8))
    offsets=sorted(set(map(int,np.linspace(0,len(observed)-width,16))))
    probes=[(f'p{i}',observed[x:x+width]) for i,x in enumerate(offsets)]
    write_fasta(work/'probes.fasta',probes)
    refs={}; records=[]
    for donor in panel:
        if mode=='chromosome' and not donor['complete']:continue
        for ri,(name,seq) in enumerate(donor['records']):
            if len(seq)<width:continue
            key=f'd{len(refs)}'
            refs[key]={'donor':donor['id'],'group':donor['group'],'complete':donor['complete'],'seq':seq,'record':name}
            records.append((key,seq+seq[:width] if donor['complete'] else seq))
    if not records:return []
    write_fasta(work/'panel.fasta',records)
    run_map(work/'panel.fasta',work/'probes.fasta',work/'observed.paf',work/'mapping.log')
    best=defaultdict(dict)
    for line in (work/'observed.paf').read_text().splitlines():
        p=line.split('\t');probe=int(p[0][1:]);qs,qe=int(p[2]),int(p[3]);ts,te=int(p[7]),int(p[8])
        qlen=int(p[1]);tlen=int(p[6]);identity=int(p[9])/max(1,int(p[10]));coverage=(qe-qs)/qlen
        if coverage<.5:continue
        score=identity*coverage
        key=(p[5],p[4])
        if probe in best[key] and best[key][probe]['score']>=score:continue
        if p[4]=='-':
            true_length=len(refs[p[5]]['seq'])
            ts,te=true_length-te,true_length-ts
        best[key][probe]={'score':score,'identity':identity,'coverage':coverage,'start':ts-qs,
                          'end':te+(qlen-qe),'clip':qs+qlen-qe}
    expected=len(observed)*missing_fraction/(1-missing_fraction)
    result=[]
    for (ref,strand),hits in best.items():
        last=len(probes)-1
        if 0 not in hits or last not in hits:continue
        a,b=hits[0],hits[last];r=refs[ref];seq=r['seq'] if strand=='+' else reverse_complement(r['seq']);n=len(seq)
        start,end=a['start'],b['end']
        if a['clip']>width*.1 or b['clip']>width*.1:continue
        observed_span=(end-start)%n if r['complete'] else end-start
        if not .5*len(observed)<=observed_span<=1.5*len(observed):continue
        if mode=='chromosome':
            length=(start-end)%n
            if not .25*expected<=length<=2.5*expected:continue
            predicted=circular_slice(seq,end,length)
        else:
            length=int(round(expected))
            if r['complete']:
                if length>n:continue
                predicted=circular_slice(seq,end,length)
            else:
                if not 0<=end<end+length<=n:continue
                predicted=seq[end:end+length]
        hs=list(hits.values());other=len(best.get((ref,'-' if strand=='+' else '+'),{}))
        x=[len(hits)/len(probes),float(np.mean([h['identity'] for h in hs])),min(h['identity'] for h in hs),
           float(np.mean([h['coverage'] for h in hs])),a['identity'],b['identity'],a['coverage'],b['coverage'],
           len(hits)/max(1,len(hits)+other),observed_span/len(observed),len(predicted)/expected,
           math.log10(len(observed)),sum(predicted.count(c) for c in 'ACGT')/len(predicted),(a['clip']+b['clip'])/(2*width)]
        result.append({'donor':r['donor'],'group':r['group'],'record':r['record'],'strand':strand,
                       'features':x,'sequence':predicted,'baseline_score':x[0]*x[1]*x[3],
                       'donor_start':start,'donor_end':end})
    # One strongest supported placement per genome; no access to hidden correctness.
    bydonor={}
    for c in result:
        if c['donor'] not in bydonor or c['baseline_score']>bydonor[c['donor']]['baseline_score']:bydonor[c['donor']]=c
    (work/'panel.fasta').unlink() # large temporary file; sources and PAF/provenance remain
    return sorted(bydonor.values(),key=lambda c:c['donor'])

def evaluate_candidates(hidden,cands,work):
    """Evaluator only. Inference must finish before this is called."""
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    if not cands:return []
    write_fasta(work/'hidden.fasta',[('withheld',hidden)])
    write_fasta(work/'candidates.fasta',[(f'c{i}',c['sequence']) for i,c in enumerate(cands)])
    run_map(work/'hidden.fasta',work/'candidates.fasta',work/'hidden_alignment.paf',work/'evaluation.log',False)
    metrics=score_alignments(hidden,[c['sequence'] for c in cands],work/'hidden_alignment.paf')
    for c,m in zip(cands,metrics):m['exact_complement']=c['sequence']==hidden
    return metrics

def score_alignments(hidden,sequences,paf):
    groups=defaultdict(list)
    for line in Path(paf).read_text().splitlines():
        p=line.split('\t')
        if 'tp:A:P' in p[12:]:groups[int(p[0][1:])].append(p)
    known_ref=np.isin(np.frombuffer(hidden.encode(),dtype=np.uint8),np.frombuffer(b'ACGT',dtype=np.uint8))
    denom=int(known_ref.sum());output=[]
    for i,seq in enumerate(sequences):
        tf=np.zeros(len(hidden),dtype=np.uint8);qf=np.zeros(len(seq),dtype=np.uint8)
        mismatch=insert=delete=0
        for p in groups[i]:
            cs=next((t[5:] for t in p[12:] if t.startswith('cs:Z:')),None)
            if cs is None:raise ValueError('Missing cs tag')
            tokens=CS.findall(cs)
            if ''.join(tokens)!=cs:raise ValueError('Unsupported cs token')
            t=int(p[7]);q=int(p[2]) if p[4]=='+' else len(seq)-int(p[3])
            for token in tokens:
                op=token[0];n=int(token[1:]) if op==':' else 1 if op=='*' else len(token)-1
                flag=1 if op in ':=' else 2
                if op!='+':tf[t:t+n]|=flag;t+=n
                if op!='-':
                    lo,hi=(q,q+n) if p[4]=='+' else (len(seq)-q-n,len(seq)-q)
                    if not 0<=lo<=hi<=len(seq):raise ValueError('Bad query coordinates')
                    qf[lo:hi]|=flag;q+=n
                mismatch+=n if op=='*' else 0;insert+=n if op=='+' else 0;delete+=n if op=='-' else 0
            assert t==int(p[8])
            assert q==(int(p[3]) if p[4]=='+' else len(seq)-int(p[2]))
        known_q=np.isin(np.frombuffer(seq.encode(),dtype=np.uint8),np.frombuffer(b'ACGT',dtype=np.uint8))
        exact_r=int(((tf==1)&known_ref).sum());exact_q=int(((qf==1)&known_q).sum())
        # Unknown predictions count against precision too; no reward for filling N.
        recall=exact_r/denom if denom else 0;precision=exact_q/len(seq) if seq else 0
        output.append({'hidden_known_bases':denom,'predicted_bases':len(seq),'hidden_exact_bases':exact_r,
                       'predicted_exact_bases':exact_q,'hidden_recovery':recall,'prediction_precision':precision,
                       'f1':2*recall*precision/(recall+precision) if recall+precision else 0,
                       'mismatch_events':mismatch,'insertion_events':insert,'deletion_events':delete,
                       'uncovered_hidden_bases':int(((tf==0)&known_ref).sum())})
    return output
