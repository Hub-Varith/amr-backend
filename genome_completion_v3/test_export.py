import gzip,json,tempfile,unittest,zipfile,hashlib
from pathlib import Path
import numpy as np
from references import read_cached,sketch_hash
from setup_references import prepare_archive
from shared import ROOT,base

class ExportTests(unittest.TestCase):
    def test_model_checksums(self):
        m=json.loads((ROOT/'EXPORT_PROVENANCE.json').read_text())
        for n,digest in m['model_sha256'].items():self.assertEqual(base.sha(ROOT/'models'/n),digest)
        for section in ['unchanged_runtime_sha256','gate_runtime_sha256']:
            for n,digest in m[section].items():self.assertEqual(base.sha(ROOT/n),digest)
    def test_restore_gzip_zip_and_reject_changed_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);seq=''.join(np.random.default_rng(42).choice(list('ACGT'),1000100))
            p=root/'truth.fasta';base.write_fasta(p,[('chromosome',seq)])
            row={'id':'synthetic','species':'ECOLI','group':'synthetic','prepared_header':'chromosome','length':len(seq),'prepared_sha256':base.sha(p),'sketch_sha256':sketch_hash(sorted(base.sketch([seq])))}
            raw=('>chromosome test\n'+seq+'\n>plasmid test\nACGT\n').encode()
            gz=root/'source.gz';gz.write_bytes(gzip.compress(raw));z=root/'source.zip'
            with zipfile.ZipFile(z,'w') as f:f.writestr('ncbi_dataset/data/test_genomic.fna',raw)
            for i,archive in enumerate([gz,z]):
                row['prepared_header']=['r0','chromosome'][i]
                base.write_fasta(p,[(row['prepared_header'],seq)])
                row['prepared_sha256']=base.sha(p)
                out=root/str(i);r=prepare_archive(row,archive,out);self.assertEqual(r['records'][0][1],seq)
                (out/'synthetic.fasta').write_text('>changed\nAAAA\n')
                with self.assertRaisesRegex(ValueError,'checksum'):read_cached(row,out)
            bad=dict(row,prepared_sha256='0'*64)
            with self.assertRaisesRegex(ValueError,'changed or wrong'):prepare_archive(bad,gz,root/'bad')
if __name__=='__main__':unittest.main()
