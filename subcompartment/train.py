"""Fresh training with a dynamic class count and a cell-disjoint evaluation fold."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from .data import load_data, sha256
from .model import Probe, predict


def train(data_path, output, *, holdout_fold=0, epochs=100, batch_size=512,
          head="resnet", device="cpu", seed=0):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    data = load_data(data_path)
    x = data["x"].astype(np.float32)
    y, classes = data["y"], data["classes"].tolist()
    test = data["fold"] == holdout_fold
    if not test.any() or test.all():
        raise ValueError("Need both training and held-out cells")
    counts = np.bincount(y[~test], minlength=len(classes))
    if (counts == 0).any():
        raise ValueError(f"Classes without training examples: {np.asarray(classes)[counts == 0].tolist()}")
    # Match historical normalization: equal class weight, then equal cell weight.
    normalization_weights = np.empty((~test).sum(), dtype=np.float64)
    train_y, train_roots = y[~test], data["root_id"][~test]
    for cls in range(len(classes)):
        ids = np.flatnonzero(train_y == cls)
        cells, inverse, per_cell = np.unique(train_roots[ids], return_inverse=True, return_counts=True)
        normalization_weights[ids] = len(train_y) / (len(classes) * len(cells) * per_cell[inverse])
    scaler = StandardScaler().fit(x[~test], sample_weight=normalization_weights)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = Probe(classes, scaler.mean_, scaler.scale_, head=head).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4 if head == "resnet" else .01,
                                 weight_decay=1e-5 if head == "resnet" else 1e-4)
    features = torch.from_numpy(x[~test])
    truth = torch.from_numpy(train_y.astype(np.int64))
    sampling = torch.from_numpy((1 / np.sqrt(counts))[train_y])
    history = []
    for epoch in range(epochs):
        model.train()
        draws = torch.multinomial(sampling, len(features), replacement=True)
        total = 0.0
        for ids in draws.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(features[ids].to(device)), truth[ids].to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        history.append({"epoch": epoch + 1, "train_loss": total / len(features)})
        print(json.dumps(history[-1]), flush=True)
    # Evaluate once after the requested epoch count; never select on the holdout.
    logits = predict(model, x[test])
    pred = logits.argmax(1)
    report = classification_report(y[test], pred, labels=np.arange(len(classes)),
                                   target_names=classes, output_dict=True, zero_division=0)
    metrics = {"macro_f1": float(f1_score(y[test], pred, labels=np.arange(len(classes)), average="macro", zero_division=0)),
               "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
               "per_class": {c: report[c] for c in classes}, "precision_zero_division": 0,
               "holdout_fold": holdout_fold, "n_train": int((~test).sum()), "n_holdout": int(test.sum()),
               "missing_holdout_classes": [classes[i] for i in range(len(classes)) if i not in set(y[test])],
               "evaluation_policy": "Fixed epochs, fresh initialization, cell-disjoint holdout evaluated once"}
    checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  "classes": classes, "embedding_dim": 64, "head": head,
                  "epochs": epochs, "seed": seed, "sampling": "inverse_sqrt",
                  "source_dataset_sha256": sha256(data_path), "holdout_fold": holdout_fold}
    for key in ("embedding_run_id", "embedding_checkpoint_id", "embedding_checkpoint_sha256"):
        if key in data:
            checkpoint[key] = str(data[key])
    torch.save(checkpoint, output / "model.pt")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    np.savez_compressed(output / "predictions.npz", sample_id=data["sample_id"][test],
                        y_true=y[test], y_pred=pred, logits=logits, classes=np.asarray(classes))
    return metrics
