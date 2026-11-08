# Historical-v117 subcompartment deployment

The 2026-11-08 run classifies **all 213,346,372 canonical embeddings from 13,713
historical roots** using all five folds of the latest excluded-cell six-class
ResNet. It covers 154,388,239 `chunkedgraph_l2_v117` nodes and 58,958,133
`cave_skeleton_v4` vertices. There is no training-positive-list inference mask.

Run directory:
`/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/historical117_subcompartment_six_class_v1`.
These cluster-specific scripts require the installed `segclr_db` environment,
the shared database, and the neighboring historical-embedding pipeline.

The input pipeline's packed receipts verify every source root was imported.
Classification reads canonical database embeddings from frozen Lance version
530191, including embeddings preserved by the input importer. Staging checks
finite 64D vectors and uses complete per-root/node-set inventories to reject
missing, duplicate or out-of-range native node IDs.

GPU workers claim immutable input files from one shared pool using filesystem
locks. Every claimed batch runs all five models independently, with no averaging.
Scores retain canonical root/node IDs; the node set is explicit in the containing
partition. Logit class order is stored in `plan.json`. Output files are read back
exactly before their completion receipts are published. Workers reuse completed
files and release claims on exit. The Slurm wrapper requeues on its pre-timeout
signal; explicit cancellation is respected.

The real L40S pilot matched independently evaluated models and CPU scores on
1,395 canonical embeddings. Publication tests cover six-class payload ordering,
identity preservation, duplicate rejection and recovery after a partial append
in a real temporary Lance database.

Five separate CPU publishers import folds concurrently. Each node-set/rank group
owns disjoint roots. A write-ahead fragment ledger permits retry without duplicate
facts, rejects conflicting stored values, and checks exact labels, uncertainty
and raw logits after each write. The dependent final audit checks complete native
node identities, uniqueness, node sets, subjects, and null synapse IDs in both
fact tables. `complete.json` is written only after those checks pass.

## Current jobs and progress

The live job ledger is `jobs.json` in the run directory. GPU scoring completed
with arrays `28364430` and `28364431`. All 213,346,372 native nodes have logits
from all five folds. Import array `28366199` and dependent verification job
`28366206` supersede the initial failed import and cancelled verification.
The import fix handles Arrow's production-sized string conversion; it passed
an exact 2,391,709-row payload check and a 2,450,000-row regression test before
resubmission. Initial failure occurred before database writes.
Submission is not completion. Logs are under the run directory's `logs/`.

```bash
../gnn_classifier/segclr_db/.venv/bin/python deployment/status_historical.py
squeue -j 28366199,28366206
```

The GPU pilot is `pilot_cuda.json`; source staging is `inputs_complete.json`.
Per-file score receipts are under `scores/`, verified publication receipts under
`imports/`, and frozen runnable source under `code/`. Check `complete.json` for
successful final publication.

The separate [same-subcompartment pair audit](PAIR_FILTER.md) reads the complete
score artifacts directly and compares L2 and skeleton geometries.

## Accessing completed predictions

`prediction_runs.json` lists each fold's separate `l2` and `cave` run IDs. A
registered run may still be incomplete while workers are active; check the final
receipt before assuming cohort-wide coverage.

```python
import json
from pathlib import Path
from segclr_db import SegCLRDatabase

work = Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/historical117_subcompartment_six_class_v1')
runs = json.loads((work / 'prediction_runs.json').read_text())
db = SegCLRDatabase()
result = db.get_predictions(
    runs[0]['l2'], root_ids=864691136577473812,
    node_set_id='chunkedgraph_l2_v117', with_logits=True,
)
```

For coverage/status use the small receipt files. The current
`db.list_prediction_runs()` implementation reads the entire predictions table
for counts; use the registry alone when inspecting run metadata.

## Read scores before database publication finishes

`inference_complete.json` confirms complete scoring independently of the later
DB-import audit. The five GPU workers that actually started completed scoring;
three unused queued workers were cancelled after the pool was empty.

```python
import numpy as np
import pyarrow.parquet as pq

plan = json.loads((work / 'plan.json').read_text())
root = 864691136577473812
rank = next(r['rank'] for r in plan['roots'] if r['root_id'] == root)
paths = sorted((work / 'scores' / 'l2' / f'{rank:02d}').glob('*.parquet'))
points = pq.read_table(paths, filters=[('root_id', '=', root)],
                       columns=['root_id', 'node_id', 'fold0_logits'])
logits = np.asarray(points['fold0_logits'].combine_chunks().values).reshape(-1, 6)
labels = np.asarray(plan['logit_classes'])[logits.argmax(axis=1)]
```
