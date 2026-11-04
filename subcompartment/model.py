"""Checkpoint-compatible residual head; normalization is stored in the model."""
import numpy as np
import torch
from torch import nn


class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(64, 128)
        self.dense_layers_1 = nn.ModuleList([nn.Linear(128, 128) for _ in range(4)])
        self.dense_layers_2 = nn.ModuleList([nn.Linear(128, 128) for _ in range(4)])

    def forward(self, x):
        hidden = self.input_layer(x)
        for first, second in zip(self.dense_layers_1, self.dense_layers_2):
            hidden = hidden + second(torch.relu(first(torch.relu(hidden))))
        return hidden


class Probe(nn.Module):
    def __init__(self, classes, mean=None, scale=None, head="resnet"):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(np.zeros(64) if mean is None else mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(np.ones(64) if scale is None else scale, dtype=torch.float32))
        self.head = head
        if head == "linear":
            self.linear = nn.Linear(64, len(classes))
        elif head == "resnet":
            self.trunk = Trunk()
            self.output = nn.Linear(128, len(classes))
        else:
            raise ValueError(f"Unknown head: {head}")

    def forward(self, x):
        x = (x - self.mean) / self.scale
        return self.linear(x) if self.head == "linear" else self.output(self.trunk(x))


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Probe(checkpoint["classes"], head=checkpoint["head"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if not torch.isfinite(model.scale).all() or (model.scale <= 0).any():
        raise ValueError("Invalid normalization scale")
    return model.to(device).eval(), checkpoint


def predict(model, x, batch_size=8192):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 64 or not np.isfinite(x).all():
        raise ValueError("Expected finite embeddings with shape (N, 64)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        parts = [model(torch.from_numpy(x[i:i + batch_size]).to(device)).cpu().numpy()
                 for i in range(0, len(x), batch_size)]
    n_classes = model.linear.out_features if model.head == "linear" else model.output.out_features
    return np.concatenate(parts) if parts else np.empty((0, n_classes), dtype=np.float32)
