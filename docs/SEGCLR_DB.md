# Accessing labels through segclr_db

Checked against the live shared database on 2026-11-04.

The Ossify source labels are registered and queryable:

```python
from segclr_db import SegCLRDatabase

db = SegCLRDatabase()
versions = db.list_node_label_versions()
labels = db.get_node_labels(
    "ossify_minnie65_structured_v1",
    node_set_id="cave_skeleton_v4",
    root_ids=864691134884807418,
)
```

This example returned 7,167 rows: 4,176 axon and 2,991 dendrite. Join node
labels to embeddings using `(node_set_id, root_id, node_id)`. Do not assume row
order matches a separate embedding query.

The collection covers 2,335 cells and 12,130,814 labels, including
`not_applicable` for glia. Its registry explicitly records
`annotation_kind="model_prediction"` and `is_ground_truth=False`. It has no
separate soma class. The historical database table name `node_ground_truth`
and the `get_node_ground_truth` compatibility alias do not imply manual truth.

**The final four-/six-class training targets are not registered as a node-label
version.** They combine Ossify process labels, cell-type assignments, soma-center
samples and spatial selection rules. The live registry contained only the
Ossify collection. Classifier predictions already stored in prediction tables
are a separate artifact and do not supply these training targets.

The complete prepared training targets and embeddings remain in per-cell NPZs
on the original shared filesystem:

- Four-class: `/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/subcompartment_latest100k_geo10um_v1/data/`
- Six-class (excluded-cell rerun): `/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/subcompartment_latest100k_sixclass_geo10um_exclude_864691135587086972_v1/data/`

Each contains `soma_cells/*.npz` and `geodesic/cells/*.npz` with `x`, `y`,
`point_id`, `xyz_nm`, scalar `root_id` and scalar `split`. The original source
fold assignments and class order are included in this repository under `models/`.
The demo packages a small subset into the portable format described in
[DATA_FORMAT.md](DATA_FORMAT.md).

These paths and the database require shared-storage access; a GitHub clone does
not grant that access. The bundled demo and model inference require neither.
Soma-center IDs occupy a different namespace from CAVE skeleton node IDs, so
publishing the combined targets into the database would require preserving their
separate node identities and label provenance. No new database writes are
performed by this repository's label editor or trainer.
