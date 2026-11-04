import csv
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from subcompartment.data import edit_labels, export_labels, load_data, save_data, sha256
from subcompartment.model import load_model, predict
from subcompartment.train import train

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def test_all_original_checkpoints_and_hashes():
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    data = load_data(ROOT / "examples/demo.npz")
    for run in manifest.values():
        for record in run["models"]:
            path = ROOT / record["file"]
            assert sha256(path) == record["sha256"]
            model, cp = load_model(path)
            logits = predict(model, data["x"][:11])
            assert logits.shape == (11, len(cp["classes"]))
            assert np.isfinite(logits).all()
            np.testing.assert_allclose(logits, predict(model, data["x"][:11], batch_size=3), rtol=1e-5, atol=1e-5)


def test_label_edits_preserve_identity_and_add_class(tmp_path):
    data = load_data(ROOT / "examples/demo.npz")
    path = tmp_path / "labels.csv"
    export_labels(data, path)
    with path.open() as handle:
        rows = list(csv.DictReader(handle))[::-1]
    sid = rows[0]["sample_id"]
    rows[0]["label"] = "new_process"
    edited = edit_labels(data, rows)
    for key in data.keys() - {"classes", "y"}:
        np.testing.assert_array_equal(data[key], edited[key])
    assert edited["classes"][-1] == "new_process"
    assert edited["y"][np.flatnonzero(data["sample_id"] == sid)[0]] == 6
    with pytest.raises(ValueError, match="exactly"):
        edit_labels(data, rows[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        edit_labels(data, rows + rows[:1])


def test_prevent_cell_leakage_and_overwrite(tmp_path):
    data = load_data(ROOT / "examples/demo.npz")
    path = tmp_path / "data.npz"
    save_data(path, data)
    with pytest.raises(FileExistsError):
        save_data(path, data)
    data["fold"][0] = (data["fold"][0] + 1) % 5
    with pytest.raises(ValueError, match="crosses folds"):
        save_data(tmp_path / "bad.npz", data)


def test_train_seven_classes_and_reload(tmp_path):
    data = load_data(ROOT / "examples/demo.npz")
    # Small representative subset with all classes in training and evaluation.
    selected = np.concatenate([np.flatnonzero((data["fold"] == f) & (data["y"] == c))[:4]
                               for f in (0, 1) for c in range(6)])
    n = len(data["x"])
    data = {k: (v[selected] if v.ndim and len(v) == n else v.copy()) for k, v in data.items()}
    rows = [{"sample_id": sid, "label": str(data["classes"][y])} for sid, y in zip(data["sample_id"], data["y"])]
    for f in (0, 1):
        index = np.flatnonzero((data["fold"] == f) & (data["y"] == 2))[0]
        rows[index]["label"] = "new_process"
    data = edit_labels(data, rows)
    source = tmp_path / "seven.npz"
    save_data(source, data)
    output = tmp_path / "trained"
    metrics = train(source, output, epochs=1, batch_size=16)
    model, cp = load_model(output / "model.pt")
    assert len(cp["classes"]) == 7
    saved = np.load(output / "predictions.npz")
    np.testing.assert_allclose(predict(model, data["x"][data["fold"] == 0]), saved["logits"], rtol=0, atol=0)
    assert metrics["missing_holdout_classes"] == []
    assert cp["source_dataset_sha256"] == sha256(source)


def test_invalid_features():
    model, _ = load_model(ROOT / "models/four_class/fold0.pt")
    with pytest.raises(ValueError, match="finite embeddings"):
        predict(model, np.full((2, 64), np.nan))
    with pytest.raises(ValueError):
        predict(model, np.zeros((2, 63)))


def test_prediction_cli_checks_encoder_provenance(tmp_path):
    import subprocess
    import sys
    data = load_data(ROOT / "examples/demo.npz")
    data["embedding_checkpoint_id"] = np.asarray("wrong_checkpoint")
    source = tmp_path / "wrong.npz"
    save_data(source, data)
    output = tmp_path / "predictions.csv"
    result = subprocess.run([sys.executable, "-m", "subcompartment.cli", "predict",
                             "--model", str(ROOT / "models/four_class/fold0.pt"),
                             "--data", str(source), "--output", str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Embedding provenance mismatch" in result.stderr
    assert not output.exists()
