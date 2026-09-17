import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
root=Path.home()/'lerobot/datasets'
arrays={};meta={}
for name in ('qiaomuyu_search_dual_v1_80_fixed','qiaomuyu_search_dual_v1_80_follower_future4'):
 p=root/name; files=sorted(p.glob('data/**/*.parquet')); d=pd.concat([pd.read_parquet(f) for f in files]).sort_values(['episode_index','frame_index'])
 prefix='original' if name.endswith('fixed') else 'future4'
 for col,key in [('action','action'),('observation.state','state')]:arrays[prefix+'_'+key]=np.stack(d[col]).astype(float)
 for col in ('timestamp','episode_index','frame_index'):arrays[prefix+'_'+col]=d[col].to_numpy()
 meta[prefix]={'dataset':name,'info':json.loads((p/'meta/info.json').read_text()),'files':[{'path':str(f.relative_to(p)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]}
np.savez_compressed('/tmp/training-comparison-20260917.npz',**arrays)
Path('/tmp/training-comparison-20260917-metadata.json').write_text(json.dumps(meta,indent=2))
print({k:v.shape for k,v in arrays.items()})
