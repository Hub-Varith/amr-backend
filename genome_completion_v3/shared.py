"""Portable runtime paths; model class names retain serialization compatibility."""
import os,platform,shutil
from pathlib import Path
import baseline_core as base
ROOT=Path(__file__).resolve().parent
bundled=ROOT/'tools/minimap2-macos'
base.MM2=os.environ.get('COMPLETION_MINIMAP2') or (str(bundled) if platform.system()=='Darwin' and platform.machine()=='arm64' else shutil.which('minimap2') or 'minimap2')
class ConstantProbability:
    def __init__(self,p):self.p=float(p)
    def predict_proba(self,x):
        import numpy as np
        return np.tile([1-self.p,self.p],(len(x),1))
