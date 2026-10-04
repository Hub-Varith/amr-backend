"""Linear candidate utility learned from within-input pairwise preferences."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def features(x):
    x=np.asarray(x,dtype=float)
    # These transforms are observed-only and used by the support baseline too.
    z=np.c_[x[:,:17],np.abs(np.log(np.maximum(.001,x[:,12])))]
    # Shared coefficients plus species-specific adjustments.
    return np.c_[z,*[z*x[:,17+i:18+i] for i in range(5)]]

class PairwiseSelector:
    def __init__(self,c=.1):self.c=c
    def fit(self,groups):
        xx=[];yy=[];ww=[]
        for x,y in groups:
            z=features(x);pairs=[]
            for i in range(len(y)):
                for j in range(i):
                    gap=y[i]-y[j]
                    if abs(gap)<.002:continue
                    diff=z[i]-z[j]
                    if gap<0:diff=-diff
                    pairs.append((diff,abs(gap)))
            total=sum(w for d,w in pairs)
            for diff,w in pairs:
                xx.extend([diff,-diff]);yy.extend([1,0]);ww.extend([w/max(total,1e-9)]*2)
        if not xx:raise ValueError('No non-tied candidate pairs')
        x=np.asarray(xx);w=np.asarray(ww);w*=len(w)/w.sum()
        self.scaler=StandardScaler(with_mean=False).fit(x)
        self.model=LogisticRegression(C=self.c,fit_intercept=False,max_iter=2000,random_state=42)
        self.model.fit(self.scaler.transform(x),yy,sample_weight=w)
        self.pairs=len(yy)//2;return self
    def predict(self,x):return self.model.decision_function(self.scaler.transform(features(x)))
