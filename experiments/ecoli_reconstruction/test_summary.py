import tempfile
import unittest
from pathlib import Path
from summarize import summarize, union_length

class SummaryTests(unittest.TestCase):
    def test_overlapping_intervals_not_double_counted(self):
        self.assertEqual(union_length([(0,10),(5,15),(20,25)]),20)
        self.assertEqual(union_length([]),0)

    def test_known_alignment_and_missing_replicon(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'r').write_text('>r\n'+'A'*100+'\n>plasmid\n'+'A'*100+'\n')
            (p/'a').write_text('>a\n'+'A'*80+'\n')
            (p/'p').write_text('a\t80\t0\t80\t+\tr\t100\t10\t90\t79\t80\t60\ttp:A:P\n')
            m=summarize(p/'r',p/'a',p/'p')
            self.assertEqual(m['reference_covered_percent'],40)
            self.assertEqual(m['aligned_identity_percent'],98.75)
            self.assertEqual(m['per_reference_covered_percent']['plasmid'],0)
            (p/'p').write_text('')
            m=summarize(p/'r',p/'a',p/'p')
            self.assertEqual(m['reference_covered_percent'],0)
            self.assertIsNone(m['aligned_identity_percent'])

if __name__=='__main__': unittest.main()
