#!/usr/bin/env python3
"""Aggregate complete CREMA-D feature shards and run actor-held-out baselines."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

p=argparse.ArgumentParser();p.add_argument('--feature-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-shards',type=int,default=12);a=p.parse_args()
files=sorted(a.feature_root.glob('features-*-shard*.npz'))
if len(files)!=a.expected_shards or a.output.exists(): raise SystemExit('incomplete or immutable aggregate')
d=[np.load(f) for f in files]; ids=np.concatenate([x['pair_id'] for x in d]); roles=np.concatenate([x['role'] for x in d]); y=np.concatenate([x['integrated_vote'] for x in d]); audio=np.concatenate([x['audio'] for x in d]); video=np.concatenate([x['video'] for x in d])
mask=np.array([':' not in v for v in y]); train=(roles=='task_fit')&mask; test=(roles=='evaluation')&mask
if len(set(ids))!=len(ids) or train.sum()==0 or test.sum()==0: raise SystemExit('invalid split')
result={"schema":"conflictbench.cremad-actor-heldout-baselines.v1","singleton_train":int(train.sum()),"singleton_evaluation":int(test.sum()),"methods":{}}
for name,x in {'audio':audio,'video':video,'audio_video':np.concatenate((audio,video),axis=1)}.items():
 m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=20270918)).fit(x[train],y[train]);result['methods'][name]={'accuracy':float(accuracy_score(y[test],m.predict(x[test]))),'feature_dim':int(x.shape[1])}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
