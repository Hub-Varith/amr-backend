import gzip, hashlib, json, os, tempfile, unittest
from pathlib import Path
from prepare import prepare

class PrepareTests(unittest.TestCase):
    def test_snakemake_empty_parent_is_accepted(self):
        before=Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                source=Path(d)/'source.fastq.gz'
                with gzip.open(source,'wt') as f:f.write('@read1\nACGT\n+\nIIII\n')
                checksum=hashlib.sha256(source.read_bytes()).hexdigest()
                Path('samples.json').write_text(json.dumps({'TEST':{'local_reads':str(source),'sha256':checksum}}))
                Path('data/TEST/subsets').mkdir(parents=True)
                prepare('TEST')
                self.assertTrue(Path('data/TEST/verified.json').exists())
                self.assertEqual(json.loads(Path('data/TEST/subsets/manifest.json').read_text())['total_reads'],1)
            finally:os.chdir(before)

if __name__=='__main__':unittest.main()
