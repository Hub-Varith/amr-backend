"""Complete one observed chromosome fragment with the frozen v3 artifacts."""
import argparse,json,gzip,time,hashlib
from pathlib import Path
import joblib
from shared import ROOT,base
from core import candidates
from release_policy import rejection_reason,THRESHOLD,MINIMUM_GROUPS
from experiment import load,pick,confidence,save

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path);p.add_argument('--species',required=True,choices=['ECOLI','KPNEU','SAUR','PAER','ABAU'])
    p.add_argument('--percent',required=True,type=int);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--state',type=Path,help='Enable the calibrated incremental confidence model and preserve session state')
    p.add_argument('--fine-final-steps',action='store_true',help='With --state, use 95% and 98% before the 100% fallback')
    a=p.parse_args();cfg=json.loads((ROOT/'config.json').read_text())
    if a.percent not in cfg['percentages']+[100]:p.error('Use a trained/evaluated percentage: '+str(cfg['percentages']+[100]))
    records=base.fasta(a.input)
    if len(records)!=1:p.error('Input must contain one continuous observed chromosome fragment; do not concatenate draft contigs')
    observed=records[0][1]
    if any(c not in 'ACGTRYSWKMBDHVN' for c in observed):p.error('Input contains unsupported nucleotide symbols')
    if len(observed)<10000:p.error('This chromosome model is not validated on fragments below 10 kb')
    old=None;history=None;grid=None
    if a.state:
        from step_state import prepare_state,write_state,adaptive_probability
        try:old,history,grid=prepare_state(a.state,observed,a.species,a.percent,a.fine_final_steps)
        except ValueError as error:p.error(str(error))
        if a.percent<100 and not (ROOT/'ADAPTIVE_CONFIDENCE_FREEZE.json').exists():p.error('Incremental confidence artifacts are not ready yet')
    # Clear public output from any previous invocation before computing a new one.
    # Incremental guesses live separately, so this cannot erase their history.
    a.output.mkdir(parents=True,exist_ok=True)
    for name in ('completed.fasta','prediction.json'):
        (a.output/name).unlink(missing_ok=True)
    if a.percent==100:
        a.output.mkdir(parents=True,exist_ok=True)
        base.write_fasta(a.output/'completed.fasta',[('fully_observed_chromosome',observed)])
        result={'species':a.species,'observed_percent_supplied':100,'observed_bases':len(observed),'ambiguous_observed_bases':sum(c not in 'ACGT' for c in observed),
            'predicted_bases':0,'confidence':None,'confidence_event':'not_applicable_full_sequence_observed',
            'decision':'full_sequence_observed','status':'full_sequence_observed','result':'full_sequence_observed',
            'sequence_file':'completed.fasta','next_observed_percent':None,'observed_prefix_preserved':True,
            'missing_bases_are_inferred':False,'clinical_validation':False,
            'note':'No completion was attempted. This does not assess sequencing errors or MIC.'}
        if a.state:write_state(a.state,observed,a.species,100,a.fine_final_steps,a.output/'completed.fasta')
        save(a.output/'prediction.json',result);print(json.dumps(result,indent=2));return
    artifacts=joblib.load(ROOT/'models/rankers.joblib');cal=joblib.load(ROOT/'models/confidence.joblib')
    frozen=json.loads((ROOT/'PRETEST_FREEZE.json').read_text())
    assert frozen['models']==base.sha(ROOT/'models/rankers.joblib')
    assert frozen['confidence']==base.sha(ROOT/'models/confidence.joblib')
    panel=[r for r in load(a.species) if r['split']=='train'];a.output.mkdir(parents=True,exist_ok=True)
    start=time.time();cs=candidates(observed,a.percent,panel,a.output/'alignment',cfg['preset'],cfg['mode'],cfg['top_k'])
    r={'species':a.species,'candidates':cs};i,scores=pick(r,artifacts['models'][a.species]);prob,n=confidence(r,i,scores,cal)
    confidence_mode='stateless';artifact_sha=None
    if a.state:
        prob,n,confidence_mode,artifact_sha=adaptive_probability(r,i,scores,history)
        if old and old.get('confidence_artifact_sha256')!=artifact_sha:p.error('Confidence model changed within this incremental session')
    reason=rejection_reason(i,prob,n)
    eligible=reason is None
    if eligible:base.write_fasta(a.output/'completed.fasta',[('observed_plus_inferred',observed+cs[i]['sequence'])])
    # Unreleased guesses are needed only for checking new observations at the next
    # step. Never publish them as completed DNA or feed them to the MIC pipeline.
    internal=None
    if a.state and i is not None:
        key=hashlib.sha256((a.species+observed+cs[i]['sequence']).encode()).hexdigest()
        internal=a.state.parent/'.completion_internal'/f'{key}.fasta'
        internal.parent.mkdir(parents=True,exist_ok=True)
        base.write_fasta(internal,[('internal_unreleased_candidate',observed+cs[i]['sequence'])])
    result={'species':a.species,'observed_percent_supplied':a.percent,'observed_bases':len(observed),'ambiguous_observed_bases':sum(c not in 'ACGT' for c in observed),
        'predicted_bases':len(cs[i]['sequence']) if eligible else 0,'candidate_count':len(cs),
        'status':'accepted' if eligible else 'no_result','result':'completed_sequence' if eligible else None,
        'message':'Candidate passed the confidence gate' if eligible else 'no result',
        'sequence_file':'completed.fasta' if eligible else None,'reason':reason,
        'confidence_threshold':THRESHOLD,'minimum_local_calibration_groups':MINIMUM_GROUPS,
        'confidence':prob,'confidence_event':cal['event'],'local_calibration_groups':n,
        'confidence_mode':confidence_mode,'previous_prediction_check':history,
        'decision':'candidate_meets_empirical_threshold' if eligible else 'no_result',
        'next_action':None if eligible else 'request_more_sequence',
        'next_observed_percent':None if eligible else min(100,((a.percent//10)+1)*10),
        'selected_donor':None if i is None else cs[i]['donor'],'method':None if i is None else cs[i]['method'],
        'model':artifacts['selection'][a.species]['chosen'],'seconds':time.time()-start,
        'observed_prefix_preserved':True,'missing_bases_are_inferred':True,'clinical_validation':False,
        'limitations':['Chromosome only; plasmids excluded','Input fraction must be supplied','Confidence is not probability of an exact genome or a correct antibiotic choice']}
    if a.state:
        result['next_observed_percent']=None if eligible else grid[grid.index(a.percent)+1]
        write_state(a.state,observed,a.species,a.percent,a.fine_final_steps,internal,artifact_sha)
    save(a.output/'prediction.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
