"""Restore the exact pinned reference bank from NCBI; no training or model changes."""
import argparse,gzip,json,subprocess,tempfile,zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from references import SPECIES,cache_root,manifest,read_cached,sketch_hash
from shared import base

def prepare_archive(row,archive,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as tmp:
        tmp=Path(tmp)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                names=[n for n in z.namelist() if n.endswith('_genomic.fna')]
                if len(names)!=1:raise ValueError('Expected one genomic FASTA in NCBI archive')
                data=z.read(names[0])
        else:
            with gzip.open(archive,'rb') as f:data=f.read()
        source=tmp/'source.fasta';source.write_bytes(data)
        records=[(h,s) for h,s in base.fasta(source) if 'plasmid' not in h.lower() and len(s)>1000000]
        if len(records)!=1:raise ValueError('Expected one non-plasmid chromosome >1Mb: '+row['id'])
        prepared=tmp/'prepared.fasta';base.write_fasta(prepared,[(row['prepared_header'],records[0][1])])
        if base.sha(prepared)!=row['prepared_sha256']:raise ValueError('NCBI chromosome changed or wrong source: '+row['id'])
        sketch=sorted(base.sketch([records[0][1]]))
        if sketch_hash(sketch)!=row['sketch_sha256']:raise ValueError('Sketch mismatch: '+row['id'])
        metadata=tmp/'sketch.json';metadata.write_text(json.dumps({'sketch':sketch})+'\n')
        prepared.replace(destination/(row['id']+'.fasta'));metadata.replace(destination/(row['id']+'.sketch.json'))
    return read_cached(row,destination)
def restore(row,root):
    try:read_cached(row,root);return row['id']+' verified'
    except FileNotFoundError:pass
    # A checksum mismatch is an error, not permission to silently overwrite a changed cache.
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        dest=Path(tmp)/'download'
        urls=[row['url']]
        fallback=f"https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{row['id']}/download?include_annotation_type=GENOME_FASTA"
        if fallback not in urls:urls.append(fallback)
        errors=[]
        for url in urls:
            try:
                subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','2','--max-time','120',url,'--output',str(dest)],check=True)
                prepare_archive(row,dest,root);return row['id']+' restored'
            except (subprocess.CalledProcessError,ValueError,OSError,zipfile.BadZipFile) as error:errors.append(str(error))
        raise RuntimeError(row['id']+': '+ '; '.join(errors))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--species',nargs='+',choices=SPECIES,default=SPECIES)
    p.add_argument('--workers',type=int,choices=[1,2],default=2);p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();rows=[r for r in manifest() if r['species'] in a.species];root=cache_root()
    if a.verify_only:
        for row in rows:read_cached(row,root)
    else:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for i,message in enumerate(pool.map(lambda row:restore(row,root),rows),1):print(f'{i}/{len(rows)} {message}',flush=True)
    print(f'Verified {len(rows)} reference chromosomes in {root}')
if __name__=='__main__':main()
