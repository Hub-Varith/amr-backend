"""Bounded sequential retries. No deletion or modification of original outputs."""
from pathlib import Path
import datetime,json,os,subprocess,time,shutil,fcntl
ROOT=Path(__file__).resolve().parent
TOOLS=json.loads((ROOT/'local_tools.json').read_text())
ALIAS=Path('/tmp/amr-five-species')
assert ALIAS.resolve()==ROOT
lock=(ROOT/'retry.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
RUN=ROOT/'retries'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S');RUN.mkdir(parents=True)
state={'status':'running','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempts':[],'limits':{'rss_gib':6,'threads':1,'attempt_minutes':45,'coverage_ladder':[30,20,15]}}
def save():
 state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 tmp=RUN/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(RUN/'status.json')
 (ROOT/'RETRY_STATUS.md').write_text('# Failed-assembly retries\n\n'+f"Run: {RUN.name}\nStatus: {state['status']}\n\n"+'\n'.join(f"- {a['species']} {a['fraction']}: initial coverage {a['coverage']}x — {a['status']}" for a in state['attempts'])+'\n\nA completed process is not validated biological accuracy. Original outputs are preserved.\n')
samples=json.loads((ROOT/'samples.json').read_text());save()
try:
 for species,level,fraction in [('SAUR','01','50%'),('KPNEU','02','100%'),('ECOLI','01','50%'),('SAUR','02','100%')]:
  for coverage in [30,20,15]:
   if shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError('Less than 12 GiB free disk')
   rel=Path('retries')/RUN.name/f'{species}_{level}_cov{coverage}'
   out=ROOT/rel;out.mkdir()
   a={'species':species,'level':level,'fraction':fraction,'coverage':coverage,'status':'running','directory':str(out)};state['attempts'].append(a);save()
   cmd=[TOOLS['snakemake'],'--snakefile',str(ROOT/'Retry.smk'),'--directory',str(out),'--cores','1','--config',f"reads={ALIAS/'data'/species/'subsets'/f'subset_{level}.fastq'}",f"out={ALIAS/rel}",f"size={samples[species]['genome_size_estimate']}",f"coverage={coverage}",f"python={TOOLS['python']}",f"flye={TOOLS['flye']}"]
   a['command']=cmd
   with (out/'workflow.log').open('w') as log:
    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    while p.poll() is None:
     g=out/'guard_status.json'
     if g.exists():
      try:a['monitor']=json.loads(g.read_text())
      except json.JSONDecodeError:pass
     save();time.sleep(5)
   g=out/'guard_status.json';a['monitor']=json.loads(g.read_text()) if g.exists() else {}
   a['returncode']=p.returncode
   if p.returncode==0 and (out/'stats.json').exists():
    a['stats']=json.loads((out/'stats.json').read_text());a['status']='completed'
    a['quality_flag']='high_fragmentation_review_required' if a['stats']['contigs']>100 else 'accuracy_not_yet_validated'
    save();break
   a['status']=a['monitor'].get('status','workflow_failed');save()
   if a['status'] not in ['memory_limit_exceeded','timeout','failed']:break
 state['status']='finished' if all(any(a['species']==sp and a['level']==lv and a['status']=='completed' for a in state['attempts']) for sp,lv in [('SAUR','01'),('KPNEU','02'),('ECOLI','01'),('SAUR','02')]) else 'finished_with_failures'
except Exception as exc:
 state.update(status='stopped',error=repr(exc));raise
finally:
 save()
