import json
from pathlib import Path
import numpy as np
import pytest
pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq
from subcompartment.export_allowlist import export


def source_fixture(tmp_path):
    six, four = tmp_path / "six", tmp_path / "four"
    assignments = {str(864691134884807418 + i): i for i in range(5)}
    classes = ["soma", "dendrite", "axon", "astrocytic_process", "microglial_process", "oligodendrocytic_process"]
    for base in (six, four):
        for folder in ("soma_cells", "geodesic/cells"):
            (base / "data" / folder).mkdir(parents=True)
        for root, fold in assignments.items():
            for folder in ("soma_cells", "geodesic/cells"):
                labels = np.array([0] if folder == "soma_cells" else list(range(1, 6 if base == six else 4)), dtype=np.int8)
                np.savez(base / "data" / folder / f"{root}.npz", root_id=root,
                         split="train" if fold else "test", y=labels, point_id=np.arange(len(labels)))
        (base / "folds.json").write_text(json.dumps({"assignments": assignments,
            "summaries": [{"fold": i, "test_counts": [1]*6} for i in range(5)]}))
    (six / "data_verification.json").write_text(json.dumps({"complete": True, "classes": classes,
        "n_cells": 5, "n_points": 30, "class_counts": [5]*6}))
    return six, four


def test_exact_allowlist_and_separate_identity_namespaces(tmp_path):
    six, four = source_fixture(tmp_path)
    output = tmp_path / "result.parquet"
    receipt = export(six, four, output)
    assert receipt["complete"] and receipt["rows"] == 30 and receipt["four_class_rows"] == 20
    table = pq.read_table(output)
    rows = table.to_pylist()
    identities = {(r["root_id"], r["sample_kind"], r["point_id"]) for r in rows}
    assert len(identities) == 30
    assert all(r["node_set_id"] is None for r in rows if r["sample_kind"] == "soma_center")
    assert pq.read_table(output, filters=[("in_four_class", "=", True)]).num_rows == 20
    with pytest.raises(FileExistsError):
        export(six, four, output)


def test_reject_duplicate_point_ids(tmp_path):
    six, four = source_fixture(tmp_path)
    path = next((six / "data/geodesic/cells").glob("*.npz"))
    with np.load(path) as archive:
        data = dict(archive)
    data["point_id"][1] = data["point_id"][0]
    np.savez(path, **data)
    output = tmp_path / "bad.parquet"
    with pytest.raises(ValueError, match="Duplicate/invalid"):
        export(six, four, output)
    assert not output.exists()
