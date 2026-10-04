import contextlib, io, json, math, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from release_policy import rejection_reason
from shared import base
import predict

class GateTests(unittest.TestCase):
    def test_threshold_and_invalid_values(self):
        self.assertIsNone(rejection_reason(0,.95,10))
        self.assertIsNone(rejection_reason(0,.99,20))
        for p in [.949999,0,float('nan'),float('inf'),-1,1.1]:
            self.assertIsNotNone(rejection_reason(0,p,10))
        self.assertEqual(rejection_reason(0,.99,9),'insufficient_calibration_support')
        self.assertEqual(rejection_reason(None,.99,10),'no_candidate')

    def invoke(self,d,species,prob=.94,groups=20,index=0,percent=65,state=None,seq=None):
        seq=seq or 'ACGT'*4000
        base.write_fasta(d/'input.fasta',[('observed',seq)])
        args=['predict.py','--input',str(d/'input.fasta'),'--species',species,
              '--percent',str(percent),'--output',str(d/'out')]
        if state:args+=['--state',str(state)]
        cs=[{'sequence':'TGCA'*100,'donor':'test_donor','method':'test'}]
        with patch.object(sys,'argv',args),patch.object(predict,'load',return_value=[]),\
             patch.object(predict,'candidates',return_value=cs),\
             patch.object(predict,'pick',return_value=(index,[1.])),\
             patch.object(predict,'confidence',return_value=(prob,groups)),\
             patch('step_state.adaptive_probability',return_value=(prob,groups,'history','test')),\
             contextlib.redirect_stdout(io.StringIO()):
            predict.main()
        return json.loads((d/'out/prediction.json').read_text())

    def test_all_species_abstain_and_clear_old_public_file(self):
        for sp in ['ECOLI','KPNEU','SAUR','PAER','ABAU']:
            with self.subTest(species=sp),tempfile.TemporaryDirectory() as tmp:
                d=Path(tmp);(d/'out').mkdir();(d/'out/completed.fasta').write_text('stale')
                r=self.invoke(d,sp)
                self.assertEqual(r['status'],'no_result');self.assertEqual(r['decision'],'no_result')
                self.assertIsNone(r['result']);self.assertIsNone(r['sequence_file'])
                self.assertEqual(r['predicted_bases'],0)
                self.assertFalse((d/'out/completed.fasta').exists())
                self.assertEqual(r['next_observed_percent'],70)

    def test_boundary_acceptance_then_insufficient_support_removes_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);r=self.invoke(d,'ECOLI',prob=.95,groups=10)
            self.assertEqual(r['status'],'accepted')
            self.assertEqual(base.fasta(d/'out/completed.fasta')[0][1],'ACGT'*4000+'TGCA'*100)
            r=self.invoke(d,'ECOLI',prob=.99,groups=9)
            self.assertEqual(r['status'],'no_result');self.assertFalse((d/'out/completed.fasta').exists())

    def test_no_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);r=self.invoke(d,'ECOLI',index=None)
            self.assertEqual(r['reason'],'no_candidate')
            self.assertFalse((d/'out/completed.fasta').exists())

    def test_incremental_history_remains_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);state=d/'state.json'
            r=self.invoke(d,'ECOLI',percent=20,state=state,seq='ACGT'*3000)
            self.assertEqual(r['status'],'no_result')
            self.assertFalse((d/'out/completed.fasta').exists())
            old=json.loads(state.read_text())
            self.assertIn('.completion_internal',old['completed_path'])
            self.assertTrue(Path(old['completed_path']).exists())
            r=self.invoke(d,'ECOLI',percent=30,state=state,seq='ACGT'*4500)
            self.assertEqual(r['status'],'no_result')
            self.assertIsNotNone(r['previous_prediction_check'])
            self.assertFalse((d/'out/completed.fasta').exists())

if __name__=='__main__':unittest.main()
