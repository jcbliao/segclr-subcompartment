# Portable dataset format

Use an NPZ containing ordinary arrays only (`allow_pickle=False`).

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `x` | `(N,64)`, finite float32 | Raw SegCLR embeddings; do not standardize beforehand |
| `y` | `(N,)`, integer | Zero-based index into `classes` |
| `classes` | `(C,)`, Unicode strings | Ordered unique class names |
| `sample_id` | `(N,)`, unique Unicode strings | Stable identity for each embedding and label |
| `root_id` | `(N,)`, Unicode strings | Cell ID; all samples of a cell stay in the same fold |
| `fold` | `(N,)`, nonnegative integer | Cell-disjoint fold assignment; 0 is the default holdout |
| `xyz_nm` | optional `(N,3)`, finite float | Physical x,y,z coordinates in nanometers |
| `point_id` | optional `(N,)`, integer | Native node or soma-center identifier |
| `sample_kind` | optional `(N,)`, Unicode strings | Distinguishes skeleton nodes from soma centers |
| `embedding_run_id` | optional scalar Unicode | SegCLR run provenance |
| `embedding_checkpoint_id` | optional scalar Unicode | SegCLR checkpoint provenance |
| `embedding_checkpoint_sha256` | optional scalar Unicode | Exact encoder weight hash |

For prediction only, `x` is sufficient; include `sample_id` to retain identity and
embedding metadata to check compatibility. For editing and training, all required
keys above are needed. Use `subcompartment.data.save_data` to validate and save a
new dataset. It refuses to replace an existing file.

```python
import numpy as np
from subcompartment.data import save_data

# X, labels, root_ids and sample_ids come from your annotations.
# Assign WHOLE CELLS to folds before creating this array.
classes = np.asarray(["soma", "dendrite", "axon", "my_new_class"])
lookup = {name: i for i, name in enumerate(classes)}
data = dict(
    x=np.asarray(X, dtype=np.float32),
    y=np.asarray([lookup[label] for label in labels], dtype=np.int64),
    classes=classes,
    sample_id=np.asarray(sample_ids, dtype=str),
    root_id=np.asarray(root_ids, dtype=str),
    fold=np.asarray([cell_to_fold[str(root)] for root in root_ids], dtype=np.int64),
)
save_data("my_dataset.npz", data)
```

Use representative examples from multiple cells for every new class, including
separate held-out cells. Never split nearby points from the same cell across
training and evaluation. The trainer refuses a class with no training examples,
and reports classes absent from the holdout. A class absent from the holdout
cannot have a meaningful generalization estimate.

To remove or merge classes, rebuild the ordered `classes` array and remap every
`y` value to the new indices. The CSV importer intentionally preserves existing
classes and only appends new ones, protecting existing label indices.

The demo uses `root_id:sample_kind:point_id` as identity. Soma-center point IDs
and skeleton node IDs refer to different namespaces. Do not join them by point
ID alone. The source labels are derived targets; retain their origin when making
manual corrections. CSV edits preserve the input arrays and fold assignments;
CLI imports also record hashes of the input NPZ and label CSV.
