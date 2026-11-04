"""Data loading and training-only normalization weights for six compartments."""
import json
import os
from pathlib import Path
import numpy as np

WORK = Path(os.environ.get('SUBCOMPARTMENT_SIX_WORK', '/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/subcompartment_latest100k_sixclass_geo10um_v1'))
BASE = WORK/'data'
CLASSES = ['soma','dendrite','axon','astrocytic_process','microglial_process','oligodendrocytic_process']


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def load(variant):
    xs=[];ys=[];roots=[];splits=[];root_ids=[];indices={}
    for folder in [BASE/'soma_cells',BASE/variant/'cells']:
        for path in sorted(folder.glob('*.npz')):
            with np.load(path) as z:
                x=z['x'];y=z['y'];rid=str(z['root_id']);split=str(z['split'])
            if not len(y):continue
            if x.shape!=(len(y),64) or not np.isfinite(x).all() or not np.isin(y,np.arange(len(CLASSES))).all():
                raise ValueError(f'Invalid six-class sample arrays: {path}')
            if rid not in indices:indices[rid]=len(root_ids);root_ids.append(rid)
            xs.append(x);ys.append(y);roots.append(np.full(len(y),indices[rid],np.int32));splits.append(np.full(len(y),split=='train',bool))
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(roots),np.concatenate(splits),root_ids


def weights(y,roots):
    result=np.empty(len(y),np.float32)
    for cls in range(len(CLASSES)):
        ids=np.flatnonzero(y==cls)
        cells,inverse,counts=np.unique(roots[ids],return_inverse=True,return_counts=True)
        if not len(ids):raise ValueError(f'Missing class {CLASSES[cls]}')
        result[ids]=len(y)/(len(CLASSES)*len(cells)*counts[inverse])
    return result
