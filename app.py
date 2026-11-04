"""Local browser label editor. Run: streamlit run app.py"""
import io
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from subcompartment.data import edit_labels, load_data

st.set_page_config(page_title="SegCLR labels", layout="wide")
st.title("Explore and edit subcompartment labels")
st.write("Edit the label column, including typing new class names. Download your edited dataset to train a new classifier.")
source = st.file_uploader("Open a portable dataset (.npz)", type="npz")
source_bytes = source.getvalue() if source is not None else (Path(__file__).parent / "examples/demo.npz").read_bytes()
source_hash = hashlib.sha256(source_bytes).hexdigest()
data = load_data(io.BytesIO(source_bytes))
frame = pd.DataFrame({"sample_id": data["sample_id"], "label": data["classes"][data["y"]],
                      "root_id": data["root_id"], "fold": data["fold"]})
if "xyz_nm" in data:
    for axis, values in zip("xyz", data["xyz_nm"].T):
        frame[f"{axis}_nm"] = values
st.caption("These example labels are derived training targets, not manually verified ground truth. Coordinates are nanometers. This table does not display EM imagery.")
edited = st.data_editor(frame, disabled=[c for c in frame if c != "label"], hide_index=True,
                        column_config={"label": st.column_config.TextColumn("label", required=True)},
                        num_rows="fixed", key="labels_" + source_hash)
try:
    updated = edit_labels(data, edited.to_dict("records"))
except (ValueError, AttributeError) as error:
    st.error(str(error))
    st.stop()
st.write("Classes:", updated["classes"].tolist())
if "xyz_nm" in data:
    st.plotly_chart(px.scatter_3d(edited, x="x_nm", y="y_nm", z="z_nm", color="label",
                                hover_data=["sample_id", "root_id"], opacity=0.6), use_container_width=True)
updated["parent_dataset_sha256"] = np.asarray(source_hash)
updated["label_edits_sha256"] = np.asarray(hashlib.sha256(edited.to_csv(index=False).encode()).hexdigest())
buffer = io.BytesIO()
np.savez_compressed(buffer, **updated)
st.download_button("Download edited dataset", buffer.getvalue(), "edited.npz", "application/octet-stream")
st.download_button("Download labels CSV", edited.to_csv(index=False), "labels.csv", "text/csv")
st.code("subcompartment train --data edited.npz --output outputs/my-model --epochs 100")
st.caption("A new class needs training examples outside fold 0. Keep all samples from the same cell in the same fold. Download before closing or replacing the dataset.")
