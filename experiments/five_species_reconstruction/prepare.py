"""Download a pinned public read set, verify its checksum, and create nested subsets."""
import hashlib, json, shutil, subprocess, sys
from pathlib import Path
from subsample import subsample

def digest(path,kind):
    with path.open('rb') as f: return hashlib.file_digest(f,kind).hexdigest()

def prepare(species):
    row=json.loads(Path('samples.json').read_text())[species]
    root=Path('data')/species; root.mkdir(parents=True,exist_ok=True)
    reads=root/'reads.fastq.gz'
    expected=row.get('sha256') or row['fastq_md5'];kind='sha256' if row.get('sha256') else 'md5'
    if not reads.exists():
        if row.get('local_reads') and Path(row['local_reads']).exists():
            reads.symlink_to(row['local_reads'])
        else:
            part=reads.with_suffix('.part')
            subprocess.run(['curl','--fail','--location','--retry','3','--connect-timeout','30','--max-time','1800','--output',str(part),row['read_url']],check=True)
            if digest(part,kind)!=expected: raise ValueError('Download checksum mismatch')
            part.replace(reads)
    if digest(reads,kind)!=expected: raise ValueError('Read checksum mismatch')
    if not (root/'subsets/manifest.json').exists():
        # Never silently reuse a partially written subset directory.
        if (root/'subsets').exists():
            if any((root/'subsets').iterdir()):
                raise RuntimeError('Nonempty incomplete subsets directory; inspect before resuming')
            (root/'subsets').rmdir()  # Snakemake pre-creates output parents.
        manifest=subsample(reads,root/'subsets',[.25,.5,1],seed=42)
    else: manifest=json.loads((root/'subsets/manifest.json').read_text())
    if row.get('base_count') and manifest['total_bases']!=int(row['base_count']): raise ValueError('ENA base count does not match downloaded data')
    if row.get('read_count') and manifest['total_reads']!=int(row['read_count']): raise ValueError('ENA read count does not match downloaded data')
    (root/'verified.json').write_text(json.dumps(dict(species=species,checksum_kind=kind,checksum=expected,manifest=manifest),indent=2))
    print(species,'verified and subsampled',flush=True)

if __name__=='__main__': prepare(sys.argv[1])
