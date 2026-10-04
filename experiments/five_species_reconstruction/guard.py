"""Bound time/RSS of only the launched process tree; preserve status for unattended work."""
import argparse, datetime, json, os, signal, subprocess, time
from pathlib import Path

def tree_rss(pid):
    lines=subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,rss='],text=True).splitlines()
    rows=[tuple(map(int,line.split())) for line in lines if len(line.split())==3]
    owned={pid}
    while True:
        expanded=owned|{p for p,parent,rss in rows if parent in owned}
        if expanded==owned: break
        owned=expanded
    return sum(rss for p,parent,rss in rows if p in owned)*1024

def run(command,status,seconds=5400,max_rss=4*1024**3):
    status=Path(status);status.parent.mkdir(parents=True,exist_ok=True)
    start=time.time();peak=0; reason='running';rc=None
    proc=subprocess.Popen(command,start_new_session=True)
    def save():
        payload=dict(status=reason,pid=proc.pid,started_at=datetime.datetime.fromtimestamp(start,datetime.timezone.utc).isoformat(),elapsed_seconds=round(time.time()-start,1),peak_observed_rss_bytes=peak,returncode=rc,command=command)
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2)+'\n');tmp.replace(status)
    def stop():
        try: os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError: return
        try: proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            proc.wait()
    try:
        save()
        while proc.poll() is None:
            peak=max(peak,tree_rss(proc.pid))
            if peak>max_rss: reason='memory_limit_exceeded';stop();break
            if time.time()-start>seconds: reason='timeout';stop();break
            save();time.sleep(2)
        rc=proc.wait()
        if reason=='running': reason='completed' if rc==0 else 'failed'
    except BaseException:
        reason='monitor_error_or_interrupted';stop();rc=proc.returncode;save();raise
    save();return 0 if reason=='completed' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--status',required=True);p.add_argument('--seconds',type=float,default=5400);p.add_argument('--max-rss-gb',type=float,default=4);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    raise SystemExit(run(command,a.status,a.seconds,int(a.max_rss_gb*1024**3)))
