# Ossify subcompartment classification

## Incoming per-node deployment with newest four-class models

**Completed and independently verified on 2026-10-27.** All five folds'
319,745,910 label rows and 319,745,910 raw-logit rows are imported into
`segclr_db`. The final two-table audit found exact expected counts, unique
native node identities, correct node sets, and no non-null synapse IDs across
all ten prediction runs. The shared run's `complete.json` contains the receipt;
`prediction_runs.json` lists queryable run IDs. The local job record also
updates automatically to `state="complete"` after the audit succeeds.

`node_deploy_common.py`, `node_deploy_inputs.py` and
`node_deploy_execute.py` run the five newest four-class checkpoints on the
latest incoming cohort from `gnn_classifier`. Classes include soma, dendrite,
axon and astrocytic process. The source incoming deployment is
`incoming_presynaptic_geo10um_latest100k_v1`, with 8,125,826 biological
synapses and 8,125,810 unique registered postsynaptic-site nodes.

Presynaptic selection takes the union of embedding-covered canonical CAVE
nodes within the existing 10,000 nm geodesic fragments. One multi-source
shortest-path computation per presynaptic root replaces repeated walks from
overlapping synapses. Full geometry is traversed, including unembedded
intermediate vertices; counts of geometric vertices without embeddings are
recorded separately. Existing presynaptic anchors, source snapshots and
verified supplemental vectors are retained. Postsynaptic sites retain the
node IDs of `incoming_post_sites_1718_d4d7691a38c4`. Both populations are
deduplicated by native `(root_id,node_id)` before scoring.

Each fold predicts every selected raw node embedding independently. There
is no embedding or logit averaging. In `segclr_db`, `root_id` and
`subject_root_id` both identify the classified cell, `node_id` is native to
the recorded node set, and **`synapse_id` is null**. Each fold has separate
presynaptic and postsynaptic prediction-run IDs. Raw logits are stored in
registered class order (astrocytic process, axon, dendrite, soma); uncertainty
is predictive entropy in nats.

The pipeline uses 32 input shards per population, up to 16 presynaptic and
32 lightweight postsynaptic selectors, four persistent GPU workers that reuse the five
models, and five parallel streaming import workers. Models and prediction
runs are registered once before parallel publication. Input, score and import
receipts commit atomically. Imports append only missing own-node rows,
refuse conflicting or duplicate facts, and exactly verify labels and logits
after write. The final receipt additionally audits global database counts,
unique native node identities, correct node sets and null synapse IDs.

Scoring completed for **55,823,372 unique presynaptic nodes** and **8,125,810
unique postsynaptic nodes**, independently across all five folds. An additional
7,226 geometric vertices in the presynaptic union lack source embeddings and
are explicitly recorded rather than assigned invented predictions. Expected
publication totals are 319,745,910 label rows and the same number of logit rows.

Publication coalesces each completed rank before writing. A write-ahead ledger
records database fragment identities before each append; exact readback scans
only newly added fragments, filtering away concurrent writes from other folds.
Retries append missing facts only, reject conflicts and duplicates, and fall
back to scoped reconciliation if compaction rewrites baseline fragments. One
shared reconciliation recovers the possible partial batch from the initial
publisher; verified batches remain intact. The final global identity audit
uses one scan per fact table across all ten run IDs. The replacement import
array is `27056842` and final audit is `27056843`; the first publisher was
stopped after its verified initial rank to remove repeated legacy-table scans.

Run root:
`/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/subcompartment_four_latest100k_incoming_nodes_v1`.
`plan.json` pins source/model provenance and `prediction_runs.json` lists the
ten database run IDs. `inputs/`, `scores/`, `imports/` and `status/` report
individual stages; `complete.json` exists only after final verification.
Current Slurm IDs are in `curation/incoming_node_prediction_jobs.json`.
`submit_node_deployment.py` submits the complete chain and refuses duplicate
job records. Logs are under `logs/subcomp_nodes_*`.

Validation passed for exact geodesic-union equivalence on 120 random graphs,
inclusive boundaries, disconnected/zero-length edges, a real postsynaptic
identity check, independent-versus-vectorized five-fold scores/labels, and
real temporary-Lance partial-write recovery with duplicate/conflict rejection.

## Use Ossify labels with segclr_db

The shared database contains the version `ossify_minnie65_structured_v1` on
`cave_skeleton_v4`. Query labels directly without reading the export files:

```python
from segclr_db import SegCLRDatabase

db = SegCLRDatabase()
labels = db.get_node_labels(
    "ossify_minnie65_structured_v1", root_ids=864691134884807418,
)
# Columns include node_set_id, root_id, node_id, label, and source.
# Join embeddings by (node_set_id, root_id, node_id).
```

The registered version preserves model configuration and the export manifest
hash, with `annotation_kind="model_prediction"` and `is_ground_truth=False`.
`import_ossify_labels.py` validates source hashes and exact vertex IDs and
coordinates before writing disjoint batches concurrently. Each batch is read
back and compared exactly; existing matching rows are reused on a retry.
Import provenance, staged batches, and the final verification receipt live at
`/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/ossify_node_labels_v1`.
The receipt is written only after every batch and the total row count pass.
Completed 2026-10-27 with four workers: 2,335 cells and 12,130,814 labels
(8,355,046 axon, 3,618,621 dendrite, 157,147 not applicable). All source node
IDs and coordinates matched the shared database, and every imported label
passed exact readback both before and after indexing.

`ossify_labels.py` reads the existing `ossify_compartments/all_cells` export
(2,335 cells). Run from this directory, or add this directory to your Python
path. No Ossify installation or inference is needed to read the labels:

```python
from ossify_labels import load_ossify_labels, ossify_labels_for_embeddings

labels = load_ossify_labels(864691134884807418)
# root_id, node_set_id, node_id, x_nm, y_nm, z_nm, ossify_code, ossify_label
```

To obtain labels in the exact row order of a `segclr_db` embedding result:

```python
from segclr_db import SegCLRDatabase

db = SegCLRDatabase()
embeddings = db.get_embeddings(
    "resnet_860b_reshuffled",
    root_ids=864691134884807418,
    checkpoint_id="checkpoint_e0_s95000",
    return_coords=True,
)
labels = ossify_labels_for_embeddings(embeddings)
axon_embeddings = embeddings.embeddings[labels.ossify_label.eq("axon").to_numpy()]
```

Codes are 2 = axon, 3 = dendrite, and 0 = not applicable (glia). These are
Ossify predictions; soma is not a separate class. The helper checks source
payload hashes and exact float32 coordinates for every requested vertex,
requires `cave_skeleton_v4`, and rejects missing cells or mismatched geometry.
For aggregated embeddings the label describes the center node, not the whole
aggregation window. The database's existing CAVE `compartment` field is a
different source. This helper reads external files; the database query above
uses the imported collection.

Validation on 2026-10-27 checked 24,916 vertices across five saved CAVE export
snapshots, shuffled/repeated node alignment, and rejection of invalid node
IDs, coordinates, absent coordinates, and other node sets. After the shared
file permissions were corrected, live shared-store reads also verified exact
coordinate equality for all 24,916 vertices in these five cells. An end-to-end
`get_embeddings(..., return_coords=True)` and label-helper check passed for
root `864691134884807418`: 7,167 embeddings, with 4,176 axon and 2,991 dendrite
labels aligned to their database nodes.

## Newest SegCLR retraining (2026-10-27)

Six-class extension: `glia_subcompartment.py` adds `microglial_process` (class
4; 30 cells) and `oligodendrocytic_process` (class 5; 16 cells) to the four
existing classes. Added process vertices must be strictly farther than
10,000 nm from their cell's `segclr_db` soma coordinates. All 46 cells have
database centers, including the oligodendrocytes missing from the older
local nucleus cache. The last user clarification specifies soma exclusion;
no synapse-distance exclusion is added. OPC and the previously excluded MC
root remain excluded.

The extension reads the same newest checkpoint directly from `segclr_db` and
verifies cached geometry against registered content hashes. Preparation
blocks on missing features, invalid centers, or conflicting vectors. The
previous four-class data are reused exactly through links in a separate
dataset. Original cell folds remain fixed; new glia in the manifest test
split enter fold 0, and their training cells are stratified by type over
folds 1–4 using seed 0. Every train/test fold must contain all six classes.

The six-output ResNet uses the established architecture, optimizer,
inverse-square-root sampler, normalization, batch size and 100 epochs.
`train_compartment_six.py` is a separate class-count adaptation of the frozen
four-class trainer, with `six_class_data.py` handling six-class weights.
New artifacts are under
`/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/subcompartment_latest100k_sixclass_geo10um_v1`.
Jobs: preparation `27039751`, five-fold training `27039764`, final summary
`27039765`; see `curation/sixclass_training_jobs.json`. Submission does not
imply completion. Separate stdout/stderr logs are under `logs/subcomp_six_*`.

Six-class results are complete: **2,331 cells and 12,122,378 samples**, including
13,930 microglial and 5,954 oligodendrocytic process samples. Mean held-out
macro F1 across five folds is **95.699%** (fold SD 1.536 percentage points),
and mean balanced accuracy is **94.557%**. Mean per-class F1 is 99.916% soma,
99.659% dendrite, 99.831% axon, 99.720% astrocytic process, 95.380%
microglial process, and 79.686% oligodendrocytic process. Averaging only the
original four class F1 scores in the six-class model gives **99.782%**,
compared with **99.815%** for the four-class baseline. Most errors in both
added classes are predictions of axon. Held-out folds also select the best
epoch, so these are checkpoint-selection scores.

The executed [six-class results notebook](six_class_results.ipynb) verifies
all five checkpoint hashes, unchanged original cell fold assignments, and
pooled confusion counts against dataset support. It includes fold and class
tables, the four-class baseline comparison, confusion plots, and existing
glial error-viewer links. Exported CSV tables and PNG/PDF figures are in
`plots/six_class_results/`. Run it with the `.venv-umap` Python environment.

Exclusion rerun: `exclude_microglia_experiment.py prepare` creates a separate
dataset omitting root `864691135587086972` from training and evaluation.
The remaining 2,330 cells retain their original folds and exact input arrays;
12,121,515 samples remain, including 13,067 microglial process samples.
`submit_excluded_microglia.py` submitted five 100-epoch folds as array
`27126678` and dependent summary/comparison job `27126679`.
Submission is not completion. The job record is
`curation/sixclass_exclude_864691135587086972_jobs.json`; artifacts are in
`/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/subcompartment_latest100k_sixclass_geo10um_exclude_864691135587086972_v1`.
The summary writes `complete.json` and `comparison_previous.json` after all
folds finish. Scores still use held-out folds for epoch selection; comparison
with the original run changes both the training and evaluation cohorts.

The four-class baseline completed successfully: all five folds ran 100 epochs, all checkpoint hashes
match their completion receipts, and all 12,102,494 samples passed dataset
verification. Mean held-out macro F1 is **99.815%**, with mean balanced
accuracy **99.812%**. Pooled held-out precision is 99.887% for soma,
99.677% for dendrite, 99.850% for axon, and 99.854% for astrocytic process.
Each cell appears in exactly one held-out fold. The held-out folds also
select the best epoch, so these are checkpoint-selection scores. Full fold
metrics, class support, and model paths are recorded in the run's
`complete.json` and `curation/latest100k_training_jobs.json`.

`latest_subcompartment.py` refreshes the original **geodesic 10 µm** dataset
with the October 22 SegCLR run
`resnet50_1280b_slab1280_50um_mac_partial_aug_2_style_bf16_100k__20261022_140504`,
checkpoint `checkpoint_e0_s100000`. The five cell-disjoint folds are copied
from `../gnn_classifier/results/subcompartment_geodesic_10um_cv/folds.json`.
The original 2,285 cells and 12,102,494 sample rows, labels, point IDs,
coordinates, and exclusion rules are preserved. Both soma and process
vectors use the new checkpoint. Existing datasets and models remain intact.

`audit_subcompartment_db.py` reads the pinned checkpoint's embeddings directly
from `segclr_db`, verifies all 2,285 cached CAVE skeletons against the database
geometry hashes, and checks other embedding point sets by exact cell and
coordinates. The successful audit read embedding table version 509460,
geometry manifest version 24647 and nodes version 61612, with no read errors
or conflicting vectors. It found **11,923,139 requested embeddings** already
in the database, including **every process point**. The remaining **179,355
points are soma centers**. `database_audit/report.json` records the evidence.

The preliminary local-archive lookup had overstated the missing count by
1,012,023 points. `refresh_subcompartment_db_plan.py` replaced that plan with
the directly retrieved database features and 22 request shards containing
only the absent soma centers. The previous plan is retained in
`archive_only_plan/`. Future submissions require the direct database audit
and refresh before inference. Unreadable geometry or conflicting vectors
block refresh rather than being treated as missing coverage.

Missing soma vectors are generated locally using the existing pinned SegCLR
inference implementation, two-channel preprocessing, CLAHE EM and
version-1718 segmentation. A GPU comparison against cached vectors must pass
before inference. The plan records checkpoint/source/code hashes, database
versions and archive file hashes. Assembly verifies every non-feature array
against the previous dataset and checks complete, finite feature coverage
before releasing training.

Training reuses `train_compartment_gpu.py` unchanged: a 64-input, 128-wide,
four-block ResNet with four output classes, no BatchNorm or dropout,
inverse-square-root sampling, training-only weighted standardization, AdamW
at lr 1e-4 / weight decay 1e-5, 16,384-point batches and 100 epochs. Each
fold selects its best checkpoint by held-out macro F1, as before; these are
checkpoint-selection scores rather than independent final test scores.

Run root:
`/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/subcompartment_latest100k_geo10um_v1`.
Prepared data live in `data/`; models, histories, predictions and metrics in
`models/fold{0..4}/models_resnet_inverse_sqrt_100epochs/geodesic/`.

The October 28 UMAP refresh uses `refresh_compartment_umaps.py` to replace
only the embeddings in each original 40,000-point UMAP sample, matching the
verified `vectors.parquet` by exact cell and coordinates. Original point IDs,
coordinates, labels, splits and row order are preserved and checked against
the saved outputs. All three fits retain the original unsupervised settings
(15 neighbors, min_dist 0.1, Euclidean distance, 500 epochs, seed 0), using
raw features without standardization. Outputs are under the run root's
`umap_same_points/geometric/geometric/`,
`umap_same_points/geodesic/geodesic/`, and
`umap_same_points/geodesic_exclusion20um/geodesic/`. Each contains
`compartment_umap.png`, `compartment_umap.pdf`, `coordinates.npz`,
`reducer.joblib`, and `settings.json`; its parent contains the refreshed
`input_samples.npz` and checkpoint/source hashes in `provenance.json`.
The new PNGs and PDFs are also copied alongside each original plot as
`compartment_umap_latest100k.{png,pdf}`. The original plots remain at their
existing paths. Run with
`/home/jcbliao/.conda/envs/segclr/bin/python -u refresh_compartment_umaps.py`;
fitting uses `.venv-umap/bin/python`. Logs for the October 28 run are
`logs/umap_same_points_latest100k.{out,err}`.

`data_verification.json` certifies preparation; top-level `complete.json`
is written only after all five folds finish and includes macro F1, balanced
accuracy and per-class precision with support (zero precision for an empty
prediction denominator).

Submitted pipeline: planning `27030145`, direct database audit `27032206`,
updated GPU preprocessing check `27032437` (superseding `27030146`),
missing-feature array `27030147`, assembly `27030148`, training array
`27030149`, final summary `27030150`. Submission does not imply that training
or verification has completed.
`curation/latest100k_training_jobs.json` records IDs and dependencies.
Each dependent job requires successful completion of its predecessor.
Separate stdout/stderr files are under `logs/subcomp_latest_*`.

`submit_latest_subcompartment.py` submits the chain and refuses to duplicate
an existing job record. The embedding array uses 16 shards, up to four
concurrently; five training folds likewise run up to four concurrently.
Embedding shards are reusable after interruption. Inspect failed stages and
the job record before submitting replacement dependencies. Generation was
held during the coverage correction and released with a dependency on the
updated GPU check; training still uses array `27030149`.
The first planning attempt `27029649` failed on shared geometry permissions;
its cancelled dependencies and failure are retained in
`curation/latest100k_training_jobs_failed_27029649.json`.

## Full TEASAR and CAVE subcompartment layers

The 2,442 registered TEASAR cells and their CAVE counterparts are exported
separately by `prepare_full_subcompartment_export.py` and
`infer_export_subcompartments.py`. The frozen inventory, pinned checkpoint,
per-vertex predictions, payload checksums, and job IDs are under
`/orcd/scratch/orcd/013/jcbliao/neuroglancer/microns/segclr_subcompartments/geodesic_exclusion20um_v1`.
Job IDs are also recorded in `curation/full_subcompartment_export_jobs.json`.

Both layers use the best geodesic 20 µm exclusion ResNet checkpoint with
inverse-square-root sampling. Labels are model predictions: soma (1, yellow),
axon (2, red), dendrite (3, blue), astrocytic process (4, green), and unknown
(0, gray). These are not Casey ground-truth labels. Native CAVE embeddings
cover all 12,358,378 vertices. TEASAR uses its own complete 111 nm densified
geometry and its own SegCLR embeddings, preserving the original cable and
vertex prefix. Of 2,442 TEASAR cells, 2,289 had cached embeddings and 153 need
new inference at 6,602,165 vertices. New embeddings are saved locally under
the export directory without modifying the shared database. Source SegCLR
run is `resnet_860b_reshuffled__20260603_150412`, checkpoint
`checkpoint_e0_s95000`, materialization 1718.

The GPU array requests up to six concurrent tasks. Tasks 0–15 use
`mit_normal_gpu` (the account quota permits four at once); tasks 16–31 use
`mit_preemptable` with `mit_general`/`normal`, moved in place on 2026-10-05 at
the user's request. Requeue is enabled and completed embedding files are
validated and reused. These tasks compute missing TEASAR embeddings. Each collection's
classifier export uses 16 CPU array shards (up to eight concurrent, four
workers per shard). Cached TEASAR cells and CAVE classify independently of
the missing-embedding jobs. Dependent jobs finish TEASAR and finalize both
collections only after all preceding tasks succeed. Interrupted jobs resume
from validated cell artifacts; failures block finalization.

Finalization checks all inputs, prediction and payload hashes, class mapping,
geometry and float32 attributes, including independent CloudVolume decoding.
Only then does it publish `skeletons/info`, `segment_properties/info`, per-layer
`neuroglancer_layer.json` and `summary.json`, and a combined `viewer_state.json`
and `viewer_url.txt`. The final Slurm job checks public HTTP byte equality and
CORS for metadata and sample payloads. Progress:

```bash
gnn_py=../gnn_classifier/segclr_db/.venv/bin/python
"$gnn_py" full_subcompartment_export_status.py
squeue -u jcbliao
```

The two resulting sources are
`precomputed://https://g-ffa18e.d1c26e.5898.data.globus.org/microns/segclr_subcompartments/geodesic_exclusion20um_v1/teasar/skeletons`
and
`precomputed://https://g-ffa18e.d1c26e.5898.data.globus.org/microns/segclr_subcompartments/geodesic_exclusion20um_v1/cave/skeletons`.
The CAVE source was finalized and public HTTP/CORS verified independently on
2026-10-05; its verification is `cave_http_verification.json`. The TEASAR source
was published with 2,290 verified skeletons on 2026-10-05; public HTTP/CORS
checks are in `teasar_http_verification.json`. Its remaining 152 exports will
be added after the outstanding embedding inference and dependent
classification succeed. The combined viewer contains both layers.
`jobs.json` and the status command report the actual run state.

To publish an initial TEASAR layer before the full run finishes, use
`infer_export_subcompartments.py finalize --kind teasar --only-ready --output
<export-directory>`. This validates and publishes only existing completed
records, lists pending roots in `teasar/summary.json`, and preserves the full
plan. The dependent full finalization later expands the same source and
segment-properties inventory without changing its URL.

Cloned https://github.com/ceesem/ossify at commit
`a028cd7b40660283e756d60a40f7bb6419d2eedc` (Ossify 0.2.6).
The editable installation means changes in `ossify/src/ossify/compartments.py`
are used immediately by the environment.

```bash
module load miniforge/25.11.0-0
mamba activate ossify
```

To recreate from this directory: `mamba env create -f environment.yml`.
`environment.yml` records the setup recipe; `requirements-lock.txt` records
the installed Python packages, including the absolute editable checkout path.

## Skeleton compatibility

- **L2 CAVE skeletons:** directly supported by `ossify.load_cell_from_client`.
  Use `synapses=True, restore_properties=True`. The default classifier requires
  graph properties `area_nm2`, `size_nm3`, `max_dt_nm`, graph-to-skeleton
  linkage, and `pre_syn` / `post_syn` annotations. The raw service skeleton
  includes `lvl2_ids` and `mesh_to_skel_map`; coordinates and edges alone are
  insufficient. Full graph edges (`restore_graph=True`) are not required for
  the default feature extraction.
- **Our new/named skeletons:** the neighboring `segclr_db_named_skeletons_pr`
  stores coordinates in nanometers, edges, and optional radii/compartments.
  Geometry can be converted using `Cell().add_skeleton(vertices=sk.coords,
  edges=sk.edges, root=soma_vertex_index)`. A segmentation root ID is not a
  skeleton root vertex index. To use bundled classifiers, we also need a valid
  mapping from L2 properties and synapses onto this new skeleton. The cached
  geometry schema alone does not retain that information, even for cached
  CAVE skeletons.
- **Prediction transfer:** the shipped classifiers aggregate over 15 downstream
  skeleton hops. Changing skeleton sampling changes the physical neighborhood
  and feature distributions. Execution compatibility does not establish model
  accuracy on a different skeletonization. Validate on matched cells; consider
  classifying the L2 skeleton first and mapping predictions to the new skeleton,
  or training on features extracted on the new skeletons.

The bundled Minnie65 and V1dd models predict **axon versus dendrite only**.
Finer subcompartments require labeled training data and a new estimator.
The current `ProbaVertexModel`, custom feature specifications (including
`ComputedFeature` for embeddings), `StructuredLabeler`, and `TransitionSchema`
already support multiclass prediction with tree constraints. This is a suitable
framework for extending to soma/basal/apical/axon, but those pretrained classes
are not provided.

Legacy binary baseline for an authenticated CAVE client and a root ID in the
matching dataset (`AxonLabel` is deprecated; prefer `ProbaVertexModel` plus
`StructuredLabeler` with explicit transition priors for new development):

```python
import ossify
from ossify.compartments import AxonLabel

cell = ossify.load_cell_from_client(
    root_id, client, synapses=True, restore_properties=True,
)
is_axon = AxonLabel("minnie65_ds15_us0_bd0").predict_axon_mask(cell)
```

Validation uses the repository's offline compartment and structured prediction
tests, including its V1dd meshwork/reference outputs. The subsequent export
below uses live CAVE data; our new skeletons have not been classified.

Setup validation: imports and both bundled model files are available;
`pip check` passes. Running `test_compartments.py`, `test_compartment_label.py`,
and `test_structured_prediction.py` produced **70 passed, 1 failed**. The failure
is the legacy vertex probability ratio reference comparison at `1e-9` tolerance:
23/7861 values differ, with maximum absolute difference among violations
`1.57e-7`. Feature parity, final segment labels, both axon masks, and the
structured prediction tests pass. The numerical discrepancy remains unresolved;
upstream code/tests were not modified. Full output: `ossify-tests.log`.

## Precomputed compartment export

`export_compartments.py` exports the 467 root IDs from the previous
`segclr_predictions/fold_test_validation/active_plan_l2.json` into a sibling
Globus directory:

```
/orcd/scratch/orcd/013/jcbliao/neuroglancer/microns/ossify_compartments/fold_test_validation
```

Run or resume with `python export_compartments.py --workers 4`. Completed
payloads are checksum-verified and reused. CAVE cells with L2 geometry and
synapses are cached in `cache/`. CAVE uses the latest valid timestamp for each
historical root; the actual timestamp is recorded per cell. The run uses the
bundled Minnie65 model through `ProbaVertexModel` and `StructuredLabeler`.
There is no `root_is_soma` argument or soma-root constraint: the root can be
either dendrite or axon. The explicit transition schema penalizes axon-to-dendrite
changes by 10 natural-log units; other changes cost 0. This is an example-based
configuration choice, not a fitted transition penalty. Probability smoothing
uses alpha 0.5, unary clipping uses 100, and optional small-region cleanup is
disabled. The complete reloadable configuration is `structured_labeler.json`.
This replaces the legacy segment averaging and root-component postprocessing.
Records from an older configuration are recomputed automatically.

The source has float32 `radius` (nm) and `compartment` attributes:

| Code | Meaning | Viewer color |
| --- | --- | --- |
| 0 | Not applicable: astrocyte, microglia, oligodendrocyte, OPC | Gray |
| 2 | Predicted axon | Red |
| 3 | Predicted dendrite | Blue |

These are binary model predictions, not ground truth. Soma is not separately
classified. The geometry is the full CAVE v4 skeleton, not the embedding-covered
L2 graph subset used by the earlier GNN prediction layer. IDs match the earlier
export, but vertex indices between the two sources must not be treated as equal.

`labels.json` records model provenance, cell types, timestamps, label counts,
and payload hashes. `viewer_state.json` and `neuroglancer_layer.json` are written
after all cells succeed. Failures are recorded in `export_errors.json`; the
initial source's `info` is published only once all requested skeletons are validated.
Updates replace payloads atomically per cell and refresh source metadata when
the full set succeeds. The original legacy export is backed up locally in
`previous_root_is_soma_false/`.

Public skeleton source:
`precomputed://https://g-ffa18e.d1c26e.5898.data.globus.org/microns/ossify_compartments/fold_test_validation/skeletons`

The export contains **467 cells (442 neurons, 25 glia), 2,424,534 vertices**.
`export-verification.json` records the most recently verified label counts.
`verify_export.py` checks all payloads using independent CloudVolume decoding,
geometry/attribute checks, label counts, SHA-256 hashes, and the current
prediction configuration. It also checks public HTTP responses and CORS for
metadata, viewer state, and one skeleton payload.

The structured rerun completed successfully (Slurm job `22786424`): all 467
files verified, public access checked, and 57 structured-prediction tests passed.

Current counts: 1,673,961 axon; 720,623 dendrite; 29,950 not applicable.
Compared with the original legacy export, 8,899 vertices across 437 cells changed
labels (6,659 dendrite-to-axon, 2,240 axon-to-dendrite). Geometry and radius bytes
are identical. See `structured-label-changes.json` and `structured-22786424.log`.

[Open the viewer](https://spelunker.cave-explorer.org/#!%7B%22dimensions%22%3A%7B%22x%22%3A%5B1e-09%2C%22m%22%5D%2C%22y%22%3A%5B1e-09%2C%22m%22%5D%2C%22z%22%3A%5B1e-09%2C%22m%22%5D%7D%2C%22position%22%3A%5B734400.0%2C616512.0%2C849440.0%5D%2C%22projectionScale%22%3A150000%2C%22layers%22%3A%5B%7B%22type%22%3A%22segmentation%22%2C%22name%22%3A%22Ossify%20axon%20%2F%20dendrite%20%28CAVE%20L2%29%22%2C%22source%22%3A%22precomputed%3A%2F%2Fhttps%3A%2F%2Fg-ffa18e.d1c26e.5898.data.globus.org%2Fmicrons%2Fossify_compartments%2Ffold_test_validation%2Fskeletons%22%2C%22segments%22%3A%5B%22864691134919403530%22%5D%2C%22skeletonRendering%22%3A%7B%22shader%22%3A%22%2F%2F%20Red%3A%20axon%3B%20blue%3A%20dendrite%3B%20gray%3A%20not%20applicable%20%28glia%29.%5Cnvoid%20main%28%29%20%7B%5Cn%20%20float%20c%20%3D%20prop_compartment%28%29%3B%5Cn%20%20if%20%28c%20%3C%201.0%29%20%7B%20emitRGB%28vec3%280.6%2C%200.6%2C%200.6%29%29%3B%20%7D%5Cn%20%20else%20if%20%28c%20%3C%202.5%29%20%7B%20emitRGB%28vec3%280.95%2C%200.20%2C%200.15%29%29%3B%20%7D%5Cn%20%20else%20%7B%20emitRGB%28vec3%280.15%2C%200.45%2C%200.95%29%29%3B%20%7D%5Cn%7D%22%2C%22mode2d%22%3A%22lines_and_points%22%2C%22mode3d%22%3A%22lines%22%2C%22lineWidth3d%22%3A2%7D%7D%5D%2C%22layout%22%3A%223d%22%7D).
It starts with one L4IT cell selected; select other exported root IDs in the
segmentation layer. `neuroglancer_layer.json` can also be added to an existing
viewer state. Browser rendering has not been visually inspected.

## All manifest cells

`run_all_cells_array.sh` runs the same pretrained binary classifier on all
2,335 cells in `gnn_classifier/data/manifest.json` (1,868 train and 467 test),
deduplicated into one `ossify_compartments/all_cells` output. Eight array tasks
run eight worker processes each. The 467 existing
test-cell skeletons and records are copied into the new output before the run;
`export_compartments.py` checks their hashes and prediction settings before
reusing them. Set `OSSIFY_EXPORT_SCOPE=all_cells` for manual export or
verification. A rerun resumes from completed cell records. The dependent
`finalize_all_cells_array.sh` job assembles metadata, then verifies all payloads
and public HTTP access after every shard succeeds.

## Synapse-based soma surface (E-I)

`E-I/` is cloned from https://github.com/isebenius/E-I at commit
`5719fe10a5df9245c609b9637303cf974ec55eba`. `estimate_microns_soma.py`
loads its standalone `soma_geometry.py` without importing unrelated pipelines.
No mesh is fetched. The runner queries postsynaptic positions in **nanometers**
and joins `synapse_target_predictions_ssa_v2.target_id` to the synapse `id`.
These are Pedigo soma/shaft/spine predictions, separate from our Ossify
axon/dendrite classifications.

Run in the `ossify` environment:

```bash
python estimate_microns_soma.py --root-id 864691134919403530 --version 1718
```

By default the center comes from `cache/<root_id>.osy` at
`cell.skeleton.vertices[cell.skeleton.root]`. Use `--center-nm X Y Z` to supply
a known soma root location instead. A skeleton root must represent the soma
for this use; arbitrary skeleton roots are unsuitable. Historical root IDs
must be queried at a compatible materialization version. Without `--version`,
the live query uses the latest version, which may have no synapses for older
IDs. Cached skeleton metadata and queried materialization version are recorded
separately; using a cached center does not synchronize the two timestamps.

For an existing labelled synapse table:

```bash
python estimate_microns_soma.py --root-id ROOT_ID \
  --synapses labelled_synapses.parquet --center-nm X Y Z
```

CSV is also supported. Required columns: `compartment` and
`post_pt_position_x`, `post_pt_position_y`, `post_pt_position_z` in nm.
If `post_pt_root_id` is present, rows are filtered to the requested cell;
otherwise provide only that cell's input synapses.

Each cell's `soma_surfaces/<root_id>/` contains `summary.csv`, the input
`synapses_nm.csv`, `provenance.json`, and `surface.npz` (kernel and harmonic
vertices in nm and triangular faces). Parameters exposed by the runner are
`--bandwidth-deg`, `--sh-degree`, `--radial-quantile`, and `--offset-nm`.
The importable `estimate_soma_surface` function exposes the full upstream API;
`summary()` returns a one-row DataFrame.

The estimated region includes soma-labelled untapered proximal trunks.
Review coverage and shaft-inside QC before interpreting measurements; the
upstream minimum of four soma synapses is an execution threshold, not a
validation of measurement quality. This produces surfaces and measurements;
it does not change the existing skeleton compartment export.

Pilot at version 1718, root `864691134919403530`: 51 soma synapses,
kernel area 351.04 um², volume 612.08 um³, harmonic area 347.73 um²,
coverage 0.7705, and shaft-inside fraction 0.0. The supplied cached root
coordinate is `[734400, 616512, 849440]` nm. A synthetic 10 um sphere
reproduced analytic area and volume to within 1%.

### Full-manifest soma bounds and SegCLR window centers

Submit `sbatch run_soma_dataset.sh`: 16 workers, 48 GB, four hours on
`mit_preemptable`, with requeue enabled. Each finished cell has an atomic
`status.json`; restarted runs reuse complete cells and retry failures.
The source is `../gnn_classifier/data/manifest.json` (2,335 cells), using its
label materialization version 1718. Glia (126) and axon-only thalamocortical
entries (17) are explicitly skipped; 2,192 entries are eligible for fitting.
The existing cached skeleton root xyz is used as the supplied center.

`run_soma_dataset.py` places window centers on a soma-centered 2,000 nm lattice
and retains centers inside the kernel radial surface. Window dimensions are
129 voxels at 32 x 32 x 40 nm, or 4,128 x 4,128 x 5,160 nm. Coverage is checked
against a 500 nm interior lattice and the fitted surface's 2,562 vertices;
uncovered probes get additional centers just inside the radial surface.
This validates sampled coverage of the fitted model, not continuous coverage
or membership in the true segmented soma. Low angular coverage remains a QC
concern and is retained in the measurements. No segmentation/mesh is fetched
for seeding, and no SegCLR inference is launched by this job.

Per-cell results live in `soma_dataset/<root_id>/`: `surface.npz`,
`summary.csv`, `provenance.json`, `synapses_nm.csv`, `seeding.json` (bounds and
coverage checks), `seed_points.csv`, `seed_points_nm.npy`, `status.json`, and
`fit.log`. Combined `soma_dataset/seed_points.csv`, `summary.csv`, and
`run_summary.json` are written after all entries are processed. CSV coordinates
are x/y/z in nm, with root IDs, local point IDs, and manifest train/test splits.
Failed cells have no successful status and are not included in combined seeds.

Initial submission: job `24606992`. The pilot cell `864691134919403530` yielded
80 centers, all inside the model, with 100% coverage over 7,465 validation probes.

### Verify soma window centers against segmentation

`sbatch validate_soma_centers.sh` checks all original seed centers with 16
workers on `mit_preemptable`. Segmentation is read at mip 0 (8 x 8 x 40 nm),
using floor(xyz_nm / voxel_resolution) for the center voxel. Reads are grouped
by storage chunk within each cell. CloudVolume's `agglomerate_cutout` resolves
the sampled labels in a batch to roots at the version-1718 timestamp. Missing
storage chunks cause a recorded validation failure, not a background label.

Original seeds are retained. Per-cell outputs include
`segmentation_center_audit.csv` (resolved roots and voxel coordinates),
`seed_points_segmentation_verified.csv`, the corresponding `_nm.npy`,
`seed_points_segmentation_rejected.csv`, `segmentation_status.json`, and
`segmentation_uncovered_probes_nm.npy`. Coverage is rechecked after filtering
on the same 500 nm interior lattice plus surface vertices. Gaps are reported;
no replacement centers are invented to cover model regions that may lie
outside the real segmentation. The run produces combined
`seed_points_segmentation_verified.csv`, `segmentation_summary.csv`, and
`segmentation_run_summary.json`. Completed validation statuses allow resume.

Pilot root `864691134919403530`: 78/80 centers belong to the intended root,
2 rejected, no uncovered validation probes after filtering.

### SegCLR inference at verified soma centers

`infer_soma_centers.py` uses the existing MICrONS crop/model implementation in
`../gnn_classifier/scripts/infer_postsynaptic_sites.py`. It runs the fixed
`resnet_860b_reshuffled` checkpoint `checkpoint_e0_s95000` (64 dimensions),
129³ crops, materialization-1718 target-root masks, EM / 255 normalization,
FP16 inference, and adaptive GPU batches starting at 128. Point IDs retain
their per-cell meaning and are separate from skeleton vertices or synapses.

Prepared inputs and output parquet shards live at
`/orcd/scratch/orcd/013/jcbliao/segclr/soma_center_embeddings_v1718`.
The manifest records checkpoint/input hashes and four rank assignments, with
exactly 90,773 centers each for normal GPU and preemptable partitions.
180 shards of at most 1,024 centers commit atomically after vector roundtrip
checks. Crop failures are retried; incomplete shards remain pending for resume.
Original seed coordinates, root IDs, local point IDs, and splits accompany
vectors. These outputs are standalone artifacts; no skeleton embedding table
is modified. Successful shards are reused on requeue.

Jobs: normal GPU array `24615515` (ranks 0–1), preemptable GPU array `24615516`
(ranks 2–3), each task with one L40S GPU, 12 CPU cores, and 48 GB RAM.
Dependent verification job `24615517` checks all point identities, coordinates,
vector dimensions, finite values, and total coverage, writing `summary.json`.
Two real input crop checks passed before submission, including mask membership
at the crop centers. No completed GPU inference is implied by submission.

### Four-class linear classifier

User-confirmed classes are soma, dendrite, axon, and astrocytic process. Use
all applicable manifest cells: astrocytes supply process samples, and
thalamocortical cells supply axon only. Microglia/oligo/OPC and the previously
excluded no-soma-synapse MC root `864691136364187874` are excluded.

`prepare_compartment_training.py` checks original embedding vertex IDs and
exact float32 xyz against the corresponding Ossify skeleton before combining
labels and features. For neurons, fitted soma membership is marked on the
source skeleton before class-boundary exclusion; soma training samples come
from the new segmentation-verified center embeddings. Process selection uses
the largest axon component before spatial exclusions, excludes both endpoints
of differing-class edges, requires distance >10 um from the nucleus point,
and excludes axon/dendrite points <=5 um from the opposite class. Two variants
use geometric and weighted-skeleton geodesic transition distances. Source
labels remain predictions rather than manually verified ground truth.

`train_compartment_linear.py` fits StandardScaler and a linear one-vs-rest
logistic classifier (SGDClassifier, L2 alpha=1e-4, 20 fixed epochs, no hidden
layers). Each class gets equal training weight, and each cell gets equal
weight within its class. Scaling uses weighted training samples only. All
samples from a cell retain the original manifest train/test split. No test
metrics are used to select parameters; evaluation occurs once after fitting.
Metrics include point accuracy, balanced accuracy, macro F1, confusion matrix,
and cell/class-balanced metrics. Model save/load prediction parity is checked.

Data and models live at
`/orcd/scratch/orcd/013/jcbliao/segclr/subcompartment_linear_v1`.
Jobs: preparation `24619984` (8 CPU workers), dependent training array
`24619985` (geometric and geodesic, CPU-only). Output models and held-out
metrics are under `models/<variant>/`. Per-cell preparation audits and failures
are retained; any preparation failure blocks both training jobs.

GPU linear training supersedes the queued CPU training at the user's request.
`train_compartment_gpu.py` fits one `nn.Linear(64,4)` with weighted softmax
cross-entropy, AdamW (lr 0.01, weight decay 1e-4), and 20 fixed epochs. There
are no hidden layers. It retains the same samples, weighted training-only
standardization, equal-class/equal-cell weighting, and held-out cell split.
Training embeddings are GPU-resident when memory permits; otherwise batches
are transferred from CPU memory. Output includes `model.pt`, raw-feature
`linear_coefficients.npz`, epoch loss history, test metrics and predictions.
Models are under `models_gpu/<variant>/` in the same scratch output directory.
This uses multinomial logistic regression in place of the cancelled CPU
one-vs-rest SGD formulation. CPU checks passed for folded normalization and
loss gradients; save/load score parity is checked after actual training.
The CPU preparation job was replaced on the preemptable partition to avoid
its normal-partition queue. Current job IDs are in `curation/training_jobs.json`.

Epoch evaluation and logging now follow the user's requested GNN convention.
All prior top-level `.log` files were moved into `logs/`. Training and dataset
preparation wrappers specify separate Slurm `.out` and `.err` files. tqdm
progress explicitly writes to stderr; per-epoch loss, full-test macro F1,
per-class F1, and best-epoch/F1 records print to stdout with flushing.

The existing manifest's whole 20% test split is evaluated every epoch, with
no new validation split. Highest unweighted macro F1 selects `model.pt`;
ties retain the earlier epoch. Best-checkpoint metrics, coefficients, and
predictions all use this selected checkpoint. Because the test split selects
the epoch, its score is a selection score rather than an independent final
holdout score. Historical final-epoch-only models are preserved in
`models_gpu_initial_final_epoch/`. Updated training array: `24621491`.
Logs: `logs/compartment_gpu_24621491_<task>.out` and `.err`.

Class-balanced sampling run: array `24621929`. Each 16,384-point training
batch contains exactly 4,096 samples from each of the four classes, sampled
with replacement. Within a class, the sampling probabilities give every cell
equal expected total frequency. The cross-entropy loss is unweighted, avoiding
a second class weighting on top of balanced sampling. Each epoch draws a
whole number of batches covering approximately the original training sample
count. Weighted training-only normalization remains unchanged.
Outputs are isolated under `models_gpu_class_balanced/<variant>/`, preserving
the prior weighted-loss models. Full-test per-epoch F1 evaluation, best-macro-F1
checkpoint selection, and separate `.out`/`.err` logs remain in place.

100-epoch class-balanced runs: GPU array `24622030`, both geometric and
geodesic variants. `train_compartment_gpu.py --epochs 100` runs from the fixed
initial seed and saves separately under `models_gpu_class_balanced_100epochs/`,
preserving the 20-epoch outputs. Full-test per-epoch F1 and best-macro-F1
checkpoint selection remain unchanged; stdout/stderr logs are in `logs/`.

ResNet-head comparison: GPU array `24622249`, 100 epochs for both variants.
`train_compartment_gpu.py --head resnet` imports the established
`gnn_classifier/gnn/resnet.py::DeepResNetTrunk`: 64-dimensional input,
128-wide hidden representation, four pre-activation residual blocks, no
BatchNorm, no dropout, and a four-class linear output. It has 140,932 trainable
parameters. AdamW uses the GNN defaults (lr=1e-4, weight decay=1e-5).
Sampling, training-only standardization, full-test per-epoch F1 evaluation,
and best-macro-F1 checkpoint selection are shared with the linear runs.
Outputs are isolated under `models_resnet_class_balanced_100epochs/`; linear
models remain intact. Logs are `logs/compartment_resnet_24622249_<task>.out`
(metrics) and `.err` (tqdm). Shape, gradient, and state restoration checks
passed before submission.

Inverse-square-root ResNet comparison: array `24653118`, both variants,
100 epochs. Per-point sampling weights are 1/sqrt(training class count),
matching the GNN sampler's frequency rule. Sampling uses replacement and
uniform points within each class (no additional per-cell correction), so
expected class totals are proportional to sqrt(class count). The loss is
unweighted. Head, optimizer, training-only normalization, epoch draw count,
full-test evaluation, and best-F1 selection remain unchanged. Outputs are
isolated under `models_resnet_inverse_sqrt_100epochs/`; previous models are
preserved. Sampling proportions passed a synthetic distribution check.
Logs: `logs/compartment_resnet_sqrt_24653118_<task>.out` and `.err`.

Equal-cell-type comparison: array `24653517`, ResNet, both variants,
100 epochs. Compartment probabilities retain the inverse-square-root rule
(total probability proportional to sqrt(training point count)). Within each
compartment, represented manifest cell types receive equal total probability;
cells within each type receive equal probability, and points within each cell
are uniform. The equivalent per-point probability is computed from training
samples only. There is no additional loss weighting. Hierarchical probability
checks passed on unequal synthetic type/cell populations. Model, optimizer,
normalization, epoch draws, evaluation, and checkpoint rule remain unchanged.
Outputs: `models_resnet_equal_cell_types_100epochs/<variant>/`.
Logs: `logs/compartment_resnet_equal_types_24653517_<task>.out` and `.err`.

Cross-evaluation of the best inverse-square-root ResNet checkpoints:
`cross_evaluate_compartments.py` evaluates each model on both full manifest test
variants, without retraining or refitting normalization. Self-evaluation must
reproduce saved macro F1 before publishing results. Predictions and metrics
are saved under `cross_evaluation_inverse_sqrt/`. GPU job: `24659997`;
stdout/stderr: `logs/compartment_cross_eval_24659997.out` and `.err`.

### Linear classifier with inverse-square-root sampling

`train_compartment_linear_sqrt.sh` runs five independent GPU tasks using the
existing four-class linear trainer on the original geodesic 10 µm dataset for
100 epochs, with per-point sampling weights `1/sqrt(training class count)`.
`train_compartment_linear_sqrt_cv.py` reuses the exact five cell-disjoint folds
from the original ResNet experiment and preserves the linear baseline's
optimizer, normalization, seed, batch size, and checkpoint-selection rule.
The original SegCLR checkpoint remains unchanged. Results are isolated under
`../gnn_classifier/results/subcompartment_linear_inverse_sqrt_10um_cv/fold*/`.
Each worker rejects concurrent runs and existing output, validates class
counts, and verifies that its held-out labels and cell IDs exactly match the
corresponding ResNet predictions before writing its completion receipt.
The dependent summary writes a final `complete.json` only after all five
folds pass, including macro F1, balanced accuracy, per-class precision and
support, and paired ResNet comparisons. Zero-denominator precision is zero.
The job record is `curation/linear_inverse_sqrt_training_job.json`;
submission alone does not establish successful training. The original
single-split job 27088135 was cancelled while pending; five-fold array
27088463 and summary job 27088464 replace it.

Completed and verified on 2026-10-28: all five folds ran 100 epochs and
matched the corresponding ResNet evaluation labels and cell IDs exactly.
The summary ran in the existing CPU allocation; its queued job was cancelled.
Mean held-out macro F1 is **0.955894** (sample SD **0.001070**), compared with
**0.988612** for the matched ResNet folds. Mean balanced accuracy is
**0.972840**. Pooled precision is 0.807958 for soma (181,546 labeled samples),
0.978919 for dendrite (3,547,915), 0.990396 for axon (8,245,375), and 0.987890
for astrocytic process (127,658). Precision is zero if no samples are
predicted for a class; all four classes were predicted. The evaluation covers
12,102,494 embeddings from 2,285 cells, each held out once. These remain
checkpoint-selection scores because each held-out fold selected its epoch.
Fold 0 improves over its class-balanced linear baseline from macro F1
0.913244 to 0.955929; a five-fold class-balanced linear baseline was not run.

### 20 µm center-exclusion experiment

`prepare_soma_exclusion_experiment.py` derives a separate geodesic dataset
from the existing 10 µm dataset by retaining process points strictly farther
than 20 µm from the same nucleus coordinates. Soma embeddings, cell splits,
and all other selection rules are preserved. The source dataset is untouched.
Per-cell removal counts are saved in `exclusion_audit.json`.

Output: `/orcd/scratch/orcd/013/jcbliao/segclr/subcompartment_geodesic_exclusion20um_v1`.
Training uses `SUBCOMPARTMENT_DATA_DIR` to select this dataset, with the same
100-epoch inverse-square-root geodesic ResNet settings and checkpoint policy.
Preparation job `24689434`; dependent training job `24689445`.
Wrappers: `run_soma_exclusion20um.sh` and
`train_compartment_resnet_exclusion20um.sh`. Both save stdout/stderr in `logs/`.
The process test samples also receive the stricter cutoff, so overall test
metrics use a different sample set; the soma test samples remain identical.

UMAP job `24690947` depends on the same preparation job. The wrapper
`compartment_umap_exclusion20um.sh` uses the advanced CPU account and fits
the existing unsupervised raw-embedding UMAP on up to 10,000 points per
compartment across train/test, geodesic only. Plots, sampled embeddings,
fitted reducer, and settings are saved under the experiment's `umap/geodesic/`.
