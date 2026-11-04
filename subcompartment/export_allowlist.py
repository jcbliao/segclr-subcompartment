"""Export exact retained sample membership; never rerun spatial selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .data import sha256

SCHEMA = pa.schema([
    ("root_id", pa.string()), ("sample_kind", pa.string()),
    ("point_id", pa.int64()), ("node_set_id", pa.string()),
    ("class_id", pa.int8()), ("label", pa.string()),
    ("fold", pa.int8()), ("in_four_class", pa.bool_()),
])
KINDS = {"soma_center": 0, "skeleton_node": 1}


def identity_bytes(root, kind, points, labels, folds):
    """Fixed-width canonical rows; hash is independent of Parquet batch boundaries."""
    rows = np.empty(len(points), dtype=[("root", "<u8"), ("kind", "u1"),
                                       ("point", "<i8"), ("label", "i1"), ("fold", "i1")])
    rows["root"], rows["kind"] = root, kind
    rows["point"], rows["label"], rows["fold"] = points, labels, folds
    return rows.tobytes()


def export(source, four_class_source, output):
    source, four_class_source, output = map(Path, (source, four_class_source, output))
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = output.with_suffix(".json")
    temp = output.with_suffix(".parquet.partial")
    if any(p.exists() for p in (output, receipt_path, temp)):
        raise FileExistsError("Output or partial export exists; refusing to replace it")
    verification = json.loads((source / "data_verification.json").read_text())
    if not verification["complete"]:
        raise ValueError("Source dataset has not been verified")
    classes = verification["classes"]
    if classes != ["soma", "dendrite", "axon", "astrocytic_process", "microglial_process", "oligodendrocytic_process"]:
        raise ValueError("Expected the established six-class ordering")
    folds_doc = json.loads((source / "folds.json").read_text())
    folds = folds_doc["assignments"]
    four_folds = json.loads((four_class_source / "folds.json").read_text())["assignments"]
    counts = np.zeros(6, dtype=np.int64)
    by_fold = np.zeros((5, 6), dtype=np.int64)
    roots, four_roots, seen_four = set(), set(), set()
    inventory, tables, pending = [], [], 0
    expected_hash = hashlib.sha256()
    with pq.ParquetWriter(temp, SCHEMA, compression="zstd", compression_level=9) as writer:
        for folder, kind in [("soma_cells", "soma_center"), ("geodesic/cells", "skeleton_node")]:
            for path in sorted((source / "data" / folder).glob("*.npz")):
                with np.load(path, allow_pickle=False) as data:
                    root = str(data["root_id"])
                    points, labels = data["point_id"], data["y"]
                    if root != path.stem or root not in folds:
                        raise ValueError(f"Cell identity/fold mismatch: {path}")
                    fold = folds[root]
                    if not 0 <= fold < 5 or (str(data["split"]) == "train") != (fold != 0):
                        raise ValueError(f"Invalid split: {path}")
                    if points.ndim != 1 or labels.shape != points.shape or points.dtype.kind not in "iu" or labels.dtype.kind not in "iu":
                        raise ValueError(f"Invalid point/label arrays: {path}")
                    if len(np.unique(points)) != len(points) or (points < 0).any() or not np.isin(labels, np.arange(6)).all():
                        raise ValueError(f"Duplicate/invalid points or labels: {path}")
                    if kind == "soma_center" and np.any(labels != 0):
                        raise ValueError(f"Non-soma target in soma namespace: {path}")
                    if kind == "skeleton_node" and np.any(labels == 0):
                        raise ValueError(f"Unexpected soma target in process set: {path}")
                    # Explicitly verify that the four-class selection is an exact subset.
                    four_path = four_class_source / "data" / folder / path.name
                    mask = labels < 4
                    if four_path.exists():
                        with np.load(four_path, allow_pickle=False) as old:
                            if str(old["root_id"]) != root or four_folds[root] != fold:
                                raise ValueError("Four-class identity/fold mismatch")
                            np.testing.assert_array_equal(old["point_id"], points[mask])
                            np.testing.assert_array_equal(old["y"], labels[mask])
                        seen_four.add(str(Path(folder) / path.name))
                        four_roots.add(root)
                    elif mask.any():
                        raise ValueError(f"Four-class samples absent from original dataset: {path}")
                    roots.add(root)
                    n = len(labels)
                    digest_bytes = identity_bytes(int(root), KINDS[kind], points, labels, fold)
                    expected_hash.update(digest_bytes)
                    inventory.append({"source_file": str(Path(folder) / path.name), "rows": n,
                                      "membership_sha256": hashlib.sha256(digest_bytes).hexdigest()})
                    counts += np.bincount(labels, minlength=6)
                    by_fold[fold] += np.bincount(labels, minlength=6)
                    tables.append(pa.Table.from_pydict({
                        "root_id": [root] * n, "sample_kind": [kind] * n,
                        "point_id": points.astype(np.int64),
                        "node_set_id": ["cave_skeleton_v4" if kind == "skeleton_node" else None] * n,
                        "class_id": labels, "label": np.asarray(classes)[labels],
                        "fold": np.full(n, fold, dtype=np.int8), "in_four_class": mask,
                    }, schema=SCHEMA))
                    pending += n
                if pending >= 131072:
                    writer.write_table(pa.concat_tables(tables), row_group_size=131072)
                    tables, pending = [], 0
            print(f"Exported {folder}; cumulative rows {int(counts.sum()):,}", flush=True)
        if tables:
            writer.write_table(pa.concat_tables(tables), row_group_size=131072)
    np.testing.assert_array_equal(counts, verification["class_counts"])
    if len(roots) != verification["n_cells"] or counts.sum() != verification["n_points"] or roots != set(folds):
        raise ValueError("Export does not match the verified source population")
    expected_four = {str(p.relative_to(four_class_source / "data"))
                     for folder in ("soma_cells", "geodesic/cells")
                     for p in (four_class_source / "data" / folder).glob("*.npz")}
    if seen_four != expected_four or four_roots != set(four_folds):
        raise ValueError("Four-class inventory is incomplete")
    for summary in folds_doc["summaries"]:
        np.testing.assert_array_equal(by_fold[summary["fold"]], summary["test_counts"])
    # Independently deserialize every output row and compare the canonical hash.
    actual_hash = hashlib.sha256()
    rows_read = 0
    for batch in pq.ParquetFile(temp).iter_batches(batch_size=131072):
        values = batch.to_pydict()
        labels = np.asarray(values["class_id"])
        if values["label"] != np.asarray(classes)[labels].tolist():
            raise ValueError("Label name mismatch on readback")
        if values["in_four_class"] != (labels < 4).tolist():
            raise ValueError("Four-class membership mismatch on readback")
        expected_nodes = ["cave_skeleton_v4" if k == "skeleton_node" else None for k in values["sample_kind"]]
        if values["node_set_id"] != expected_nodes:
            raise ValueError("Node namespace mismatch on readback")
        actual_hash.update(identity_bytes(np.asarray(values["root_id"], dtype=np.uint64),
                          [KINDS[k] for k in values["sample_kind"]], values["point_id"], labels, values["fold"]))
        rows_read += len(batch)
    if rows_read != int(counts.sum()) or actual_hash.digest() != expected_hash.digest():
        raise ValueError("Exported membership does not exactly match source")
    receipt = {"complete": True, "source_experiment": source.name,
               "four_class_source_experiment": four_class_source.name,
               "rows": rows_read, "cells": len(roots), "classes": classes,
               "class_counts": counts.tolist(), "class_counts_by_fold": by_fold.tolist(),
               "four_class_rows": int(counts[:4].sum()), "four_class_cells": len(four_roots),
               "source_verification_sha256": sha256(source / "data_verification.json"),
               "source_folds_sha256": sha256(source / "folds.json"),
               "membership_sha256": actual_hash.hexdigest(), "parquet_sha256": sha256(temp),
               "bytes": temp.stat().st_size, "identity": ["root_id", "sample_kind", "point_id"],
               "source_files": inventory}
    temp.rename(output)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return {k: v for k, v in receipt.items() if k != "source_files"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Verified six-class experiment directory")
    parser.add_argument("--four-class-source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.four_class_source, args.output), indent=2))


if __name__ == "__main__":
    main()
