"""Pinned chromosome reference cache; does not read benchmark targets or labels."""
import hashlib,json,os
from pathlib import Path
from shared import ROOT,base

SPECIES=['ECOLI','KPNEU','SAUR','PAER','ABAU']
def cache_root():return Path(os.environ.get('COMPLETION_REFERENCE_DIR',str(ROOT/'cache'))).expanduser().resolve()
def manifest():return json.loads((ROOT/'references.json').read_text())['references']
def sketch_hash(values):return hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()
def read_cached(row,root=None):
    root=cache_root() if root is None else Path(root)
    seq=root/(row['id']+'.fasta');meta=root/(row['id']+'.sketch.json')
    if not seq.exists() or not meta.exists():raise FileNotFoundError(f"Missing {row['id']}. Run python setup_references.py --species {row['species']}")
    if base.sha(seq)!=row['prepared_sha256']:raise ValueError('Reference checksum mismatch: '+row['id'])
    values=json.loads(meta.read_text())['sketch']
    if sketch_hash(values)!=row['sketch_sha256']:raise ValueError('Reference sketch checksum mismatch: '+row['id'])
    records=base.fasta(seq)
    if len(records)!=1 or len(records[0][1])!=row['length']:raise ValueError('Invalid reference chromosome: '+row['id'])
    return dict(row,split='train',complete=True,records=records,sketch=set(values))
def load(species):
    if species not in SPECIES:raise ValueError('Unsupported species: '+species)
    rows=[r for r in manifest() if r['species']==species]
    return [read_cached(r) for r in rows]
