"""Observed-only state for incremental requests; never contains hidden truth."""
import json,hashlib
from pathlib import Path
import numpy as np,joblib
from shared import ROOT,base
from experiment import save,confidence_x,logit
from adaptive_confidence import history_features

def prepare_state(path,observed,species,percent,fine=False):
    grid=list(range(20,100,10))+([95,98] if fine else [])+[100]
    if percent not in grid:raise ValueError('Percentage is outside the chosen incremental schedule')
    old=json.loads(Path(path).read_text()) if Path(path).exists() else None
    if old is None:
        if percent not in [20,100]:raise ValueError('A new incremental session starts at 20% (or a full 100% input)')
        return None,[-1.,0.],grid
    if old['species']!=species:raise ValueError('Species changed within an incremental session')
    if old['fine_final_steps']!=fine:raise ValueError('Stopping schedule changed within an incremental session')
    n=old['observed_bases']
    if len(observed)<n or hashlib.sha256(observed[:n].encode()).hexdigest()!=old['observed_sha256']:
        raise ValueError('New DNA must extend the exact previous observed prefix')
    expected=grid[grid.index(old['percent'])+1] if old['percent']<100 else None
    if percent!=expected:raise ValueError(f'Next scheduled percentage is {expected}')
    if len(observed)<=n:raise ValueError('Additional sequence is required')
    guess=None
    if old.get('completed_path'):
        if base.sha(old['completed_path'])!=old['completed_sha256']:raise ValueError('Previous prediction file changed')
        previous=base.fasta(old['completed_path'])[0][1]
        if hashlib.sha256(previous[:n].encode()).hexdigest()!=old['observed_sha256']:raise ValueError('Previous prediction file changed')
        guess=previous[n:]
    return old,history_features(observed[n:],guess),grid
def adaptive_probability(row,index,scores,history):
    freeze=json.loads((ROOT/'ADAPTIVE_CONFIDENCE_FREEZE.json').read_text());path=ROOT/'models/adaptive_confidence.joblib'
    if base.sha(path)!=freeze['artifact_sha256'] or base.sha(ROOT/'models/rankers.joblib')!=freeze['rankers_sha256']:
        raise ValueError('Adaptive confidence artifacts changed')
    a=joblib.load(path);mode=a['selected'];x=confidence_x(row,index,scores)
    if x is None:return 0.,0,mode,freeze['artifact_sha256']
    if mode=='history':x=x+history
    m=a['models'][mode];raw=m['model'].predict_proba(np.asarray([x]))[:,1]
    p=float(m['calibrator'].predict_proba(logit(raw))[0,1])
    groups={r['group'] for r in a['support'][mode] if abs(r['probability']-p)<=.05}
    return p,len(groups),mode,freeze['artifact_sha256']
def write_state(path,observed,species,percent,fine,completed,artifact_sha=None):
    save(path,{'species':species,'percent':percent,'observed_bases':len(observed),
        'observed_sha256':hashlib.sha256(observed.encode()).hexdigest(),'fine_final_steps':fine,
        'completed_path':None if completed is None else str(Path(completed).resolve()),
        'completed_sha256':None if completed is None else base.sha(completed),'confidence_artifact_sha256':artifact_sha})
