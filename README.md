# SegCLR subcompartment classification

Pretrained classifiers for **64-dimensional SegCLR embeddings**, with a browser
label editor and portable CPU/GPU training. Clone this repository to run the
models, experiment with labels, or add new classes. No cluster account or database
credentials are needed for the included examples.

## Quick start

Use Python 3.10 or newer (tested with Python 3.11).

```bash
git clone https://github.com/jcbliao/segclr-subcompartment.git
cd segclr-subcompartment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e '.[ui]'

# Predict on the bundled example embeddings.
subcompartment predict --model models/six_class/fold0.pt \
  --data examples/demo.npz --output outputs/predictions.csv

# Open the browser label editor.
streamlit run app.py
```

The editor shows sample identities, cell IDs, folds, labels and 3D coordinates.
Change any label in the table, including typing a new class name, and download
`edited.npz`. You can also upload your own dataset in the same format. The editor
shows sample locations; it is not an EM-image annotation viewer. Keep cell/root
IDs as strings when opening CSVs in spreadsheets to avoid rounding 64-bit IDs.

## Edit labels and train additional classes

The browser's downloaded dataset can be used directly:

```bash
subcompartment train --data edited.npz --output outputs/my-model --epochs 100
subcompartment predict --model outputs/my-model/model.pt \
  --data examples/demo.npz --output outputs/my-predictions.csv
```

For a CSV workflow, edit only the `label` column, then import the changes:

```bash
subcompartment export-labels --data examples/demo.npz --output labels.csv
# Edit labels.csv in your preferred editor. New label strings add new classes.
subcompartment apply-labels --data examples/demo.npz \
  --labels labels.csv --output edited.npz
subcompartment train --data edited.npz --output outputs/my-model --epochs 100
```

Labels are matched by stable `sample_id`, so sorting the CSV does not scramble
embedding alignment. Missing or duplicate IDs are rejected. Existing outputs are
not overwritten. The original class order is retained and new names are appended;
every class in the dataset's class list must have training examples. If you want
to remove a class entirely, remap `y` and `classes` together (see the data format).
A new class needs correctly annotated examples, not just a name. Renaming a
pretrained checkpoint's class list does not teach it a new category.

The trainer creates a fresh model with the required number of outputs. It uses
inverse-square-root class sampling, training-only weighted normalization, and
AdamW. Fold 0 is held out by default; all samples from a cell must stay in one
fold. Evaluation happens once after the fixed epoch count. Use `--holdout-fold`,
`--head linear`, `--device cuda`, or `--epochs` to change the experiment.
The output includes `model.pt`, training history, held-out predictions, macro F1,
balanced accuracy, per-class precision/recall/F1 and support. Undefined precision
is zero; absent held-out classes are listed explicitly.

## Included models and data

| Directory | Outputs | Checkpoints |
| --- | --- | --- |
| `models/four_class/` | soma, dendrite, axon, astrocytic process | folds 0–4 |
| `models/six_class/` | the four above, microglial process, oligodendrocytic process | folds 0–4 |

All ten checkpoints are the original trained weights, including normalization,
committed directly to Git; no separate weight download or Git LFS setup is needed.
The six-class models are from the completed rerun excluding microglial root
`864691135587086972`. Each model is a 64 → 128 residual MLP with four
pre-activation blocks, no BatchNorm/dropout, and a linear output layer.
[Model hashes](models/manifest.json), fold assignments, original training settings,
metrics and completion summaries are included alongside the checkpoints.

`examples/demo.npz` contains 3,840 real embeddings (128 per class per fold),
coordinates and editable labels from the six-class dataset. Source hashes and
sampling details are in [examples/provenance.json](examples/provenance.json).
This small, class-balanced subset is for trying the workflow. It is not the full
12-million-point training set and cannot reproduce the published metrics. The
pretrained fold-0 models have seen cells from folds 1–4; running pretrained models
on the entire demo is not a held-out evaluation. For pretrained evaluation, use
only samples whose fold matches the model's fold, and note the selection caveat
below.

## Complete filtered positive list

[`data/filtered_samples.parquet`](data/filtered_samples.parquet) contains the exact
**12,121,515 retained sample identities and training labels** from the six-class
set. Its `in_four_class` flag selects the verified original four-class subset.
This 22 MB file is included in the clone and contains no embeddings. See
[data/README.md](data/README.md) for filtering by cell/fold and joining to
`segclr_db` labels or embeddings without rerunning the spatial filters.

## Embedding compatibility and interpretation

These are **classification heads**, not raw EM encoders. The included models
expect embeddings from:

- Run: `resnet50_1280b_slab1280_50um_mac_partial_aug_2_style_bf16_100k__20261022_140504`
- Checkpoint: `checkpoint_e0_s100000`
- Encoder checkpoint SHA-256: `826eebf68fdd47e979385c8f401825a8c9f4f2ea2c838794066f5b906a080897`

The raw-image encoder and full training embeddings are not included. New raw EM
images require that matching encoder and its preprocessing to generate features.
Do not mix embeddings from other SegCLR checkpoints, even if they also have 64
features. The prediction CLI rejects conflicting embedding provenance when both
the input and model provide it; an array without metadata cannot be checked.

Original targets combine Ossify-derived process labels, soma-center samples,
cell-type information and spatial exclusions. They are not manually verified
ground truth. The original five-fold runs selected their best epoch using the
held-out fold, so saved metrics are checkpoint-selection scores, not an
independent final test. New portable training evaluates once after fixed epochs;
repeatedly tuning against its holdout still turns that fold into a validation set.

## Database access

The Ossify axon/dendrite source labels are available through `segclr_db`. The
complete four-/six-class training targets are not currently registered there.
See [SEGCLR_DB.md](docs/SEGCLR_DB.md) for a verified query and full-dataset paths.

All new historical-v117 embeddings have been classified with all five six-class
folds. [Deployment details](deployment/README.md) describe the complete saved
scores and resumable database publication. The completed
[same-subcompartment pair comparison](docs/historical_pair_filter/README.md)
includes pooled error rates, retention, plots, and per-neuron results for 1,924
neurons across L2 and skeleton geometries.

## Python API and your own data

```python
import numpy as np
from subcompartment import load_model, predict

model, metadata = load_model("models/six_class/fold0.pt")
with np.load("examples/demo.npz", allow_pickle=False) as data:
    logits = predict(model, data["x"])
labels = np.asarray(metadata["classes"])[logits.argmax(axis=1)]
```

See [DATA_FORMAT.md](docs/DATA_FORMAT.md) for adding your own embeddings,
labels and cell-disjoint folds. To contribute label edits or a new experiment,
fork this public repo and submit a pull request with the label CSV and a short
explanation of the annotations and data source. Large datasets can be shared
separately with their hashes and provenance.

The original cluster-dependent preparation/training scripts are preserved in
[docs/research_scripts](docs/research_scripts); they are reference code and require
the original neighboring projects and datasets. The portable package above is
self-contained. [Labeling rules](docs/LABELING_RULES.md) describe the four-class
base; [research history](docs/RESEARCH_HISTORY.md) records the six-class extension
and historical experiments. The completed excluded-cell six-class rerun's current
status is in `models/six_class/summary.json`.

Related projects: [segclr_classifier](https://github.com/jcbliao/segclr_classifier),
[Ossify](https://github.com/ceesem/ossify), and
[E-I](https://github.com/isebenius/E-I). The residual head reproduces the
`gnn/resnet.py` architecture used by the original classifier project.

## Checks

```bash
python -m pip install -e '.[test]'
pytest -q
```
