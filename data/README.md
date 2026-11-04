# Positive list of retained training samples

[`filtered_samples.parquet`](filtered_samples.parquet) is the **complete positive
list**, exported directly from the prepared six-class dataset. Membership means
that a point passed the original filtering and was retained as a training or
held-out evaluation sample. It does not mean that the point was used to train
every model: for checkpoint fold `k`, fold `k` is held out and all other folds
are training samples.

- Six-class set: **12,121,515 samples, 2,330 cells**.
- Four-class subset (`in_four_class=True`): **12,102,494 samples, 2,285 cells**.
- Includes **181,546 soma-center samples**; the remaining rows are skeleton nodes.
- Exact identity: **`(root_id, sample_kind, point_id)`**.
- File size: approximately 22 MB. No embeddings or images are stored here.

The six-class set excludes microglial root `864691135587086972`. This is the
current excluded-cell rerun, not every historical filtering variant. The export
explicitly compared each four-class source file's point IDs, labels and cell
fold against the corresponding subset; the four-class flag is therefore a
verified membership selector, not an approximation based on a new filter.

| Column | Meaning |
| --- | --- |
| `root_id` | Cell ID as a string; preserves 64-bit identity in spreadsheet tools |
| `sample_kind` | `skeleton_node` or `soma_center` |
| `point_id` | Native ID within that cell and sample namespace |
| `node_set_id` | `cave_skeleton_v4` for skeleton nodes; null for soma centers |
| `class_id` | Integer target index in the class order below |
| `label` | The retained training target, not a classifier prediction |
| `fold` | Original cell-disjoint fold, 0–4 |
| `in_four_class` | Whether the exact sample also belongs to the four-class set |

Class order: soma, dendrite, axon, astrocytic_process, microglial_process,
oligodendrocytic_process. These are derived training targets, not manually
verified ground truth. In particular, cell-type rules can override the original
Ossify label; use `label` here to recover the actual training target.

## Read only the points you need

```bash
python -m pip install -e '.[data]'
```

```python
import pyarrow.parquet as pq

# Positive list for one cell's skeleton nodes in the original four-class set.
keep = pq.read_table(
    "data/filtered_samples.parquet",
    filters=[
        ("root_id", "=", "864691134884807418"),
        ("sample_kind", "=", "skeleton_node"),
        ("in_four_class", "=", True),
    ],
).to_pandas()
node_ids = keep["point_id"].to_numpy()
```

Drop `in_four_class` to use the six-class set. Add `("fold", "=", 0)` for
fold-0 evaluation, or `("fold", "!=", 0)` for fold-0 training. Reading the
entire file into pandas can consume much more memory than the compressed file;
use Parquet filters or `pq.ParquetFile(...).iter_batches()`.

## Filter labels obtained from segclr_db

```python
from segclr_db import SegCLRDatabase

root = 864691134884807418
db = SegCLRDatabase()
ossify = db.get_node_labels(
    "ossify_minnie65_structured_v1",
    node_set_id="cave_skeleton_v4",
    root_ids=root,
)
ossify["root_id"] = ossify["root_id"].astype(str)
selected = keep.rename(columns={"point_id": "node_id", "label": "training_label"})
filtered = selected.merge(
    ossify.rename(columns={"label": "ossify_label"}),
    on=["node_set_id", "root_id", "node_id"],
    how="left", validate="one_to_one",
)
assert filtered["ossify_label"].notna().all()
```

This retains the positive list's rows and makes missing database labels visible.
Use the same identity keys when joining embeddings; never use row position.
`training_label` and `ossify_label` have different provenance and need not agree.
Soma centers are not CAVE skeleton nodes: do not join their `point_id` to a
skeleton `node_id`. Their prepared embeddings remain in the source
`soma_cells/*.npz` files.

## Verification and regeneration

[`filtered_samples.json`](filtered_samples.json) records counts, class/fold
counts, source inventories, source verification hashes, the Parquet SHA-256,
and a canonical membership hash. Export validation checks point uniqueness
within each cell/namespace, source/fold inventories, class counts, exact
four-class subset equality, and every output row after Parquet readback.

With access to the original prepared experiment directories:

```bash
python -m subcompartment.export_allowlist \
  --source /path/to/subcompartment_latest100k_sixclass_geo10um_exclude_864691135587086972_v1 \
  --four-class-source /path/to/subcompartment_latest100k_geo10um_v1 \
  --output filtered_samples.parquet
```

The exporter reads retained IDs and labels; it does not regenerate the spatial
filters. Existing outputs are never replaced. This file supplies portable
membership; it does not register a new collection in `segclr_db`.
