"""Observed-only reference-assisted completion with coherent long alignments."""
import hashlib,json,math,subprocess,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from shared import base,ROOT

FEATURES=['fraction','containment','coherent_coverage','coherent_identity','best_full_coverage',
 'best_full_identity','left_identity','right_identity','left_clip_hidden_ratio',
 'right_clip_hidden_ratio','left_span_fraction','right_span_fraction','predicted_length_ratio',
 'reference_length_ratio','boundary_alternatives','log_observed_length','boundary_support']

def interval_quality(hits):
    if not hits:return 0.,0.
    points=sorted({v for h in hits for v in (h['qs'],h['qe'])})
    covered=weighted=0.
    for a,b in zip(points,points[1:]):
        identities=[h['identity'] for h in hits if h['qs']<=a and h['qe']>=b]
        if identities:covered+=b-a;weighted+=(b-a)*max(identities)
    return covered,weighted/max(1,covered)

def retrieve(observed,panel,top_k):
    q=base.sketch([observed]);scored=[]
    for r in panel:
        score=len(q&r['sketch'])/max(1,len(q))
        scored.append((score,r))
    scored.sort(key=lambda x:(-x[0],x[1]['id']))
    out=[];groups=set()
    for score,row in scored:
        if row['group'] in groups:continue
        out.append((score,row));groups.add(row['group'])
        if len(out)>=top_k:break
    return out

def align(observed,selected,work,preset='asm10',mode='full'):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    n=len(observed);width=min(150000,n)
    queries=[('left',observed[:width]),('right',observed[-width:])]
    offsets={'left':0,'right':n-width}
    if mode in ('full','hybrid'):queries.append(('full',observed));offsets['full']=0
    elif mode=='tiles':
        for i,start in enumerate(sorted(set(map(int,np.linspace(0,max(0,n-40000),24))))):
            name=f't{i}';queries.append((name,observed[start:start+40000]));offsets[name]=start
    else:raise ValueError(mode)
    signature={'input':hashlib.sha256(observed.encode()).hexdigest(),'preset':preset,'mode':mode,
               'donors':[[r['id'],r.get('prepared_sha256')] for _,r in selected]}
    manifest=work/'alignment_signature.json';paf=work/'observed.paf'
    if paf.exists() and manifest.exists():
        if json.loads(manifest.read_text())!=signature:raise RuntimeError('Cached alignment inputs changed')
    else:
        base.write_fasta(work/'query.fasta',queries)
        # Double circles permit long observed alignments across the FASTA origin.
        base.write_fasta(work/'references.fasta',[(f'd{i}',r['records'][0][1]*2) for i,(_,r) in enumerate(selected)])
        cmd=[base.MM2,'-x',preset,'-t','2','-c','--cs=short','--secondary=yes','-N','100','-p','0.05',
             str(work/'references.fasta'),str(work/'query.fasta')]
        start=time.time()
        commands=[]
        with (work/'observed.part').open('w') as out,(work/'alignment.log').open('w') as err:
            if mode=='hybrid':
                base.write_fasta(work/'query.fasta',queries[:2])
                subprocess.run(cmd,stdout=out,stderr=err,check=True,timeout=300);commands.append(cmd[:])
                base.write_fasta(work/'context.fasta',[('full',observed)])
                coarse=[x for x in cmd[:-1] if x not in ('-c','--cs=short')]+[str(work/'context.fasta')]
                subprocess.run(coarse,stdout=out,stderr=err,check=True,timeout=300);commands.append(coarse)
                (work/'context.fasta').unlink()
            else:
                subprocess.run(cmd,stdout=out,stderr=err,check=True,timeout=300);commands.append(cmd)
        (work/'observed.part').replace(paf)
        manifest.write_text(json.dumps(signature,indent=2)+'\n')
        (work/'timing.json').write_text(json.dumps({'seconds':time.time()-start,'commands':commands})+'\n')
        (work/'references.fasta').unlink();(work/'query.fasta').unlink()
    hits=defaultdict(list)
    for line in paf.read_text().splitlines():
        p=line.split('\t');i=int(p[5][1:]);length=len(selected[i][1]['records'][0][1]);strand=p[4]
        qs,qe=int(p[2])+offsets[p[0]],int(p[3])+offsets[p[0]]
        ts,te=int(p[7]),int(p[8])
        if strand=='-':ts,te=2*length-te,2*length-ts
        identity=int(p[9])/max(1,int(p[10]))
        if qe-qs<500 or identity<.80:continue
        hits[i,strand].append({'qs':qs,'qe':qe,'ts':ts,'te':te,'identity':identity,'query':p[0],
                              'mapq':int(p[11]),'matches':int(p[9])})
    return hits

def candidates(observed,percent,panel,work,preset='asm10',mode='full',top_k=12,max_per_donor=3):
    if not 0<percent<100:raise ValueError('Need a genuinely partial sequence')
    if not panel:return []
    rescue=mode in ('rescue','full_rescue')
    alignment_mode='full' if mode=='full_rescue' else 'hybrid' if mode=='rescue' else mode
    selected=retrieve(observed,panel,top_k);hits=align(observed,selected,Path(work),preset,alignment_mode)
    length=len(observed);expected=length*(100-percent)/percent;results=[]
    max_clip=min(25000,max(1000,expected*.20))
    for (di,strand),hh in hits.items():
        containment,donor=selected[di];seq=donor['records'][0][1]
        if strand=='-':seq=base.reverse_complement(seq)
        n=len(seq)
        left=sorted([h for h in hh if h['qs']<=max_clip and (alignment_mode!='hybrid' or h['query']!='full')],key=lambda h:(h['qs'],-h['identity'],-(h['qe']-h['qs'])))[:12]
        right=sorted([h for h in hh if length-h['qe']<=max_clip and (alignment_mode!='hybrid' or h['query']!='full')],key=lambda h:(length-h['qe'],-h['identity'],-(h['qe']-h['qs'])))[:12]
        options={}
        for a in left:
            for b in right:
                start=(a['ts']-a['qs'])%n;end=(b['te']+(length-b['qe']))%n
                span=(end-start)%n;missing=(start-end)%n
                if not missing or not .5*length<=span<=1.5*length:continue
                if not .05*expected<=missing<=4*expected:continue
                coherent=[]
                for h in hh:
                    rel=(h['ts']-start)%n
                    if rel<=span+max_clip and abs(rel-h['qs'])<=max(25000,.15*length):coherent.append(h)
                coverage,identity=interval_quality(coherent)
                full=[h for h in coherent if h['query']=='full']
                full_best=max(full,key=lambda h:h['qe']-h['qs'],default=None)
                support=min(a['qe']-a['qs'],b['qe']-b['qs'])/max(1,min(length,150000))
                x=[percent/100,containment,coverage/length,identity,
                   0 if full_best is None else (full_best['qe']-full_best['qs'])/length,
                   0 if full_best is None else full_best['identity'],a['identity'],b['identity'],
                   a['qs']/max(1,expected),(length-b['qe'])/max(1,expected),
                   (a['qe']-a['qs'])/length,(b['qe']-b['qs'])/length,missing/expected,
                   n/(length+expected),0,math.log10(length),min(1.,support)]
                score=.4*containment+.4*x[2]+.2*identity-.08*abs(math.log(missing/expected))-.1*(x[8]+x[9])
                key=(start,end)
                if key not in options or score>options[key]['baseline_score']:
                    options[key]={'donor':donor['id'],'group':donor['group'],'strand':strand,
                        'method':mode+'_'+preset,'features':x,'baseline_score':score,'sequence':base.circular_slice(seq,end,missing),
                        'start':start,'end':end,'left_clip':a['qs'],'right_clip':length-b['qe']}
        options=sorted(options.values(),key=lambda c:-c['baseline_score'])
        for c in options[:max_per_donor]:c['features'][14]=min(20,len(options))/20;results.append(c)
        if rescue:
            # Explicit lower-support alternatives: infer length from the supplied
            # fraction and anchor the suffix next to the observed cut. No truth.
            # Both left- and right-anchored alternatives permit structural variation.
            for side,anchors in [('right',right[:2]),('left',left[:2])]:
                for h in anchors:
                    missing=round(expected)
                    if missing>=n:continue
                    if side=='right':end=(h['te']+length-h['qe'])%n;start=(end+missing)%n
                    else:start=(h['ts']-h['qs'])%n;end=(start-missing)%n
                    cov,ident=interval_quality(hh)
                    full=[z for z in hh if z['query']=='full'];best=max(full,key=lambda z:z['qe']-z['qs'],default=None)
                    lc=h['qs']/expected if side=='left' else 1.
                    rc=(length-h['qe'])/expected if side=='right' else 1.
                    x=[percent/100,containment,cov/length,ident,
                       0 if best is None else (best['qe']-best['qs'])/length,
                       0 if best is None else best['identity'],h['identity'] if side=='left' else 0,
                       h['identity'] if side=='right' else 0,lc,rc,
                       (h['qe']-h['qs'])/length if side=='left' else 0,
                       (h['qe']-h['qs'])/length if side=='right' else 0,missing/expected,
                       n/(length+expected),min(20,len(options))/20,math.log10(length),0.]
                    score=.4*containment+.4*x[2]+.2*ident-.1*(lc+rc)
                    results.append({'donor':donor['id'],'group':donor['group'],'strand':strand,
                        'method':'length_'+side+'_'+preset,'features':x,'baseline_score':score,
                        'sequence':base.circular_slice(seq,end,missing),'start':start,'end':end,
                        'left_clip':None if side=='right' else h['qs'],
                        'right_clip':None if side=='left' else length-h['qe']})
    bysequence={}
    for c in results:
        key=hashlib.sha256(c['sequence'].encode()).hexdigest()
        if key not in bysequence or c['baseline_score']>bysequence[key]['baseline_score']:bysequence[key]=c
    return sorted(bysequence.values(),key=lambda c:(-c['baseline_score'],c['donor']))
