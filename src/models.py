"""GNN and baseline model definitions for Na-ion voltage prediction."""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool
from torch_geometric.data import Data


class CGCNN(nn.Module):
    """Crystal Graph Convolutional Neural Network for voltage prediction."""

    def __init__(self, node_dim=92, edge_dim=41, hidden_dim=128, n_conv=3, n_fc=2):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([
            CGConv(hidden_dim, dim=edge_dim, batch_norm=True)
            for _ in range(n_conv)
        ])
        fc_layers = []
        for i in range(n_fc):
            in_dim = hidden_dim if i == 0 else hidden_dim // 2
            out_dim = hidden_dim // 2
            fc_layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(0.2)])
        fc_layers.append(nn.Linear(hidden_dim // 2, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, data):
        x = self.node_embed(data.x)
        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index, data.edge_attr))
        x = global_mean_pool(x, data.batch)
        return self.fc(x).squeeze(-1)


def save_model(model, history, path, config=None):
    """Save PyTorch model weights and training history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    hist_path = path.with_suffix(".json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    if config:
        conf_path = path.with_name(path.stem + "_config.json")
        with open(conf_path, "w") as f:
            json.dump(config, f, indent=2)


def load_model(model, path):
    """Load PyTorch model weights."""
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def save_rf_model(model, path, config=None):
    """Save sklearn random forest model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    if config:
        conf_path = path.with_suffix(".json")
        with open(conf_path, "w") as f:
            json.dump(config, f, indent=2)
