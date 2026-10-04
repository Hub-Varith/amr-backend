"""Start one resumable Snakemake workflow; no concurrent assemblies."""
import datetime,json,os,shutil,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent;os.chdir(root)
if shutil.disk_usage(root).free<12*1024**3: raise SystemExit('Need at least 12 GiB free space')
config=json.loads((root/'local_tools.json').read_text())
(root/'logs').mkdir(exist_ok=True);(root/'status').mkdir(exist_ok=True)
lock=root/'batch.pid'
if lock.exists():
    try:os.kill(int(lock.read_text()),0)
    except ProcessLookupError:pass
    else:raise SystemExit('A batch with the saved PID is still running; refusing duplicate')
lock.write_text(str(os.getpid()))
args=[config['snakemake'],'--cores','2','--keep-going','--rerun-incomplete','--config',f"python={config['python']}",f"flye={config['flye']}",f"minimap={config['minimap']}"]
state={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'status':'running','command':args}
(root/'batch_status.json').write_text(json.dumps(state,indent=2))
with (root/'logs/workflow.log').open('a') as log:
    p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
    while p.poll() is None:
        subprocess.run([sys.executable,'report.py'],stdout=subprocess.DEVNULL,check=True)
        time.sleep(10)
state.update(status='completed' if p.returncode==0 else 'finished_with_failures',returncode=p.returncode,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
(root/'batch_status.json').write_text(json.dumps(state,indent=2))
subprocess.run([sys.executable,'report.py'],check=True)
lock.unlink(missing_ok=True)
