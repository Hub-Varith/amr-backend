"""Inference helpers exported verbatim from the frozen v3 experiment; no training entrypoint."""
import json
from pathlib import Path
import numpy as np
from shared import ROOT,base
from references import load
SPECIES=['ECOLI','KPNEU','SAUR','PAER','ABAU']
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def augmented(c,species):return c['features']+[int(species==s) for s in SPECIES]

def scores_for(r,model):
    cs=r['candidates']
    if not cs:return []
    if model=='support':return [c['baseline_score'] for c in cs]
    if model=='closest':return [c['features'][1]-.01*abs(np.log(max(.001,c['features'][12])))-.02*(c['features'][8]+c['features'][9]) for c in cs]
    return list(map(float,model.predict(np.asarray([augmented(c,r['species']) for c in cs]))))

def pick(r,model):
    scores=scores_for(r,model);idx=int(np.argmax(scores)) if scores else None
    return idx,scores

def confidence_x(r,idx,scores):
    if idx is None:return None
    c=r['candidates'][idx];ss=sorted(scores,reverse=True)
    return augmented(c,r['species'])+[scores[idx],ss[0]-ss[1] if len(ss)>1 else 0,min(100,len(scores))/100]

def logit(p):
    p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p)).reshape(-1,1)

def confidence(r,idx,scores,artifact):
    v=confidence_x(r,idx,scores)
    if v is None:return 0.,0
    raw=artifact['model'].predict_proba(np.asarray([v]))[:,1]
    p=float(artifact['calibrator'].predict_proba(logit(raw))[0,1])
    groups={z['group'] for z in artifact['support'] if abs(z['probability']-p)<=.05}
    return p,len(groups)
