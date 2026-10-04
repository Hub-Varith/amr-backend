import tempfile,unittest
from pathlib import Path
import numpy as np
from core import candidates,interval_quality
from shared import base

class Tests(unittest.TestCase):
    def test_union_not_double_counted(self):
        c,q=interval_quality([{'qs':0,'qe':100,'identity':.9},{'qs':50,'qe':150,'identity':1}])
        self.assertEqual(c,150);self.assertAlmostEqual(q,145/150)

    def test_circular_orientation_and_observed_only(self):
        rng=np.random.default_rng(24);seq=''.join(rng.choice(list('ACGT'),200000))
        donor={'id':'other_genome','group':'other_group','complete':True,'records':[('ref',base.reverse_complement(seq))],'sketch':base.sketch([seq])}
        for percent,mode in [(p,m) for p in [20,65,95,98] for m in ['full','hybrid','full_rescue']]:
            rotated=base.circular_slice(seq,157913,len(seq));cut=len(seq)*percent//100
            with tempfile.TemporaryDirectory() as tmp:
                cs=candidates(rotated[:cut],percent,[donor],Path(tmp),'asm10',mode,1)
                self.assertTrue(cs)
                self.assertEqual(cs[0]['sequence'],rotated[cut:])
                cached=candidates(rotated[:cut],percent,[donor],Path(tmp),'asm10',mode,1)
                self.assertEqual(cached,cs)
                with self.assertRaisesRegex(RuntimeError,'Cached alignment inputs changed'):
                    altered=('A' if rotated[0]!='A' else 'C')+rotated[1:cut]
                    candidates(altered,percent,[donor],Path(tmp),'asm10',mode,1)

if __name__=='__main__':unittest.main()
