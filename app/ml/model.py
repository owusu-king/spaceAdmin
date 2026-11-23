"""SpaceFunctionNet: a multilayer perceptron with learned entity embeddings.

    categorical ids ─► Embedding per column ─┐
                                              ├─► concat ─► [Linear ─ BatchNorm ─ ReLU ─ Dropout] x2 ─► Linear ─► logits
    standardized numerics ───────────────────┘

Entity embeddings let the network learn dense representations of buildings, floors,
departments and room-number shapes instead of sparse one-hot vectors.
"""
from __future__ import annotations

import torch
from torch import nn


def embedding_dim(cardinality: int) -> int:
    """Common heuristic: grows with cardinality, capped to keep the model small."""
    return int(min(16, max(2, round(1.6 * cardinality ** 0.56))))


class SpaceFunctionNet(nn.Module):
    def __init__(self, cardinalities: list[int], n_numeric: int, n_classes: int,
                 hidden: tuple[int, ...] = (128, 64), dropout: float = 0.25, use_embeddings: bool = True) -> None:
        super().__init__()
        self.use_embeddings, self.cardinalities = use_embeddings, list(cardinalities)
        # Ablation option: one-hot vectors instead of learned embeddings.
        self.embeddings = nn.ModuleList(nn.Embedding(c, embedding_dim(c)) for c in cardinalities) if use_embeddings else nn.ModuleList()
        cat_width = sum(e.embedding_dim for e in self.embeddings) if use_embeddings else sum(cardinalities)
        width = cat_width + n_numeric
        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, n_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        if self.use_embeddings:
            parts = [layer(cat[:, i]) for i, layer in enumerate(self.embeddings)]
        else:
            parts = [nn.functional.one_hot(cat[:, i], c).float() for i, c in enumerate(self.cardinalities)]
        return self.mlp(torch.cat([*parts, num], dim=1))
