import numpy as np,pytest
from app import split_indices,confidence_threshold,summarize,predict

def test_split_disjoint_and_stratified():
 y=np.repeat(np.arange(7),50);a,b,c=split_indices(y)
 assert not(set(a)&set(b) or set(a)&set(c) or set(b)&set(c))
 assert len(set(a)|set(b)|set(c))==len(y)
 assert all(len(np.unique(y[x]))==7 for x in [a,b,c])

def test_threshold_uses_validation_precision():
 p=np.array([[.9,.1],[.6,.4],[.1,.9],[.4,.6]]);y=np.array([0,1,1,0]);t=confidence_threshold(p,y);assert .6<t<=.9

def test_bad_inference_rejected():
 with pytest.raises(ValueError):predict({'features':[1,2]})
 with pytest.raises(ValueError):predict({'features':[float('nan')]*27})

def test_all_abstain():
 y=np.array([0,1]);p=np.full((2,7),1/7);s=summarize(y,p,.9);assert s['coverage']==0 and s['accepted_accuracy'] is None
