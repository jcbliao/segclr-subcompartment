import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from .data import edit_labels, export_labels, load_data, save_data, sha256
from .model import load_model, predict
from .train import train


def main():
    parser = argparse.ArgumentParser(description="Predict compartments, edit labels, and train new classes")
    parser.add_argument("--threads", type=int, default=4)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="NPZ with x=(N,64); optional sample_id and embedding metadata")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("export-labels")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("apply-labels")
    p.add_argument("--data", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--holdout-fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--head", choices=["linear", "resnet"], default="resnet")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "predict":
        model, checkpoint = load_model(args.model, args.device)
        with np.load(args.data, allow_pickle=False) as data:
            for key in ("embedding_run_id", "embedding_checkpoint_id", "embedding_checkpoint_sha256"):
                if key in data and key in checkpoint and str(data[key]) != checkpoint[key]:
                    raise ValueError(f"Embedding provenance mismatch: {key}")
            logits = predict(model, data["x"])
            ids = data["sample_id"] if "sample_id" in data else np.arange(len(logits)).astype(str)
            if ids.shape != (len(logits),) or len(set(ids)) != len(ids):
                raise ValueError("sample_id must be unique with one ID per embedding")
        classes = np.asarray(checkpoint["classes"])
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["sample_id", "predicted_label", *[f"p_{c}" for c in classes]])
            writer.writerows((sid, classes[p.argmax()], *p.tolist()) for sid, p in zip(ids, probabilities))
        print(f"Wrote {len(ids)} predictions to {path}")
    elif args.command == "export-labels":
        export_labels(load_data(args.data), args.output)
    elif args.command == "apply-labels":
        with open(args.labels, newline="") as handle:
            result = edit_labels(load_data(args.data), csv.DictReader(handle))
        result["parent_dataset_sha256"] = np.asarray(sha256(args.data))
        result["label_edits_sha256"] = np.asarray(sha256(args.labels))
        save_data(args.output, result)
        print(f"Saved classes: {result['classes'].tolist()}")
    else:
        kwargs = vars(args).copy()
        kwargs.pop("command"); kwargs.pop("threads")
        kwargs["data_path"] = kwargs.pop("data")
        print(json.dumps(train(**kwargs), indent=2))


if __name__ == "__main__":
    main()
