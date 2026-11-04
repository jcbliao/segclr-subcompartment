"""Identity-preserving label edits and a portable NPZ dataset format."""
import csv
import hashlib
from pathlib import Path
import numpy as np

REQUIRED = {"x", "y", "classes", "sample_id", "root_id", "fold"}


def load_data(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    validate(data)
    return data


def validate(data):
    if missing := REQUIRED - data.keys():
        raise ValueError(f"Missing fields: {sorted(missing)}")
    x, y, classes = data["x"], data["y"], data["classes"]
    n = len(x)
    if not n or x.shape != (n, 64) or not np.isfinite(x).all():
        raise ValueError("x must be a nonempty finite (N,64) array")
    if classes.ndim != 1 or classes.dtype.kind != "U" or len(set(classes)) != len(classes) or any(not c.strip() for c in classes):
        raise ValueError("classes must be unique nonempty strings")
    if y.shape != (n,) or y.dtype.kind not in "iu" or not np.isin(y, np.arange(len(classes))).all():
        raise ValueError("y must contain integer indices into classes")
    for key in ("sample_id", "root_id", "fold"):
        if data[key].shape != (n,):
            raise ValueError(f"{key} must have one value per sample")
    for key in ("sample_id", "root_id"):
        if data[key].dtype.kind != "U" or any(not v.strip() for v in data[key]):
            raise ValueError(f"{key} must contain nonempty strings (preserve 64-bit IDs)")
    if len(set(data["sample_id"])) != n:
        raise ValueError("sample_id must be unique")
    if data["fold"].dtype.kind not in "iu" or (data["fold"] < 0).any():
        raise ValueError("fold must contain nonnegative integers")
    assignment = {}
    for root, fold in zip(data["root_id"], data["fold"]):
        if root in assignment and assignment[root] != fold:
            raise ValueError(f"Cell {root} crosses folds")
        assignment[root] = fold
    if "xyz_nm" in data and (data["xyz_nm"].shape != (n, 3) or not np.isfinite(data["xyz_nm"]).all()):
        raise ValueError("xyz_nm must be finite (N,3) coordinates")


def save_data(path, data):
    validate(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **data)


def export_labels(data, path):
    with Path(path).open("x", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sample_id", "label", "root_id", "fold"])
        writer.writerows(zip(data["sample_id"], data["classes"][data["y"]], data["root_id"], data["fold"]))


def edit_labels(data, rows):
    """Require the complete ID set; CSV row order cannot change feature alignment."""
    rows = list(rows)
    labels = {}
    for row in rows:
        sid, label = row["sample_id"], row["label"].strip()
        if sid in labels or not label:
            raise ValueError("Duplicate sample_id or empty label")
        labels[sid] = label
    if set(labels) != set(data["sample_id"]):
        raise ValueError("Label edits must contain exactly the original sample IDs")
    classes = list(data["classes"])
    classes += sorted(set(labels.values()) - set(classes))
    result = dict(data)
    result["classes"] = np.asarray(classes)
    index = {name: i for i, name in enumerate(classes)}
    result["y"] = np.asarray([index[labels[sid]] for sid in data["sample_id"]], dtype=np.int64)
    validate(result)
    return result


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
