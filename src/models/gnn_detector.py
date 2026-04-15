"""
GNN models for ghost-artist detection.

Two architectures:
  GhostDetectorGAT — Graph Attention Network (primary model)
    Attention mechanism learns which edges matter most.
    Ghost artists connected to other ghosts receive high attention weights.
    Interpretable: attention scores can be visualized per edge.

  GhostDetectorGCN — Graph Convolutional Network (baseline)
    Standard neighbourhood aggregation, no attention.
    Faster to train; used as comparison baseline.

Both output a scalar in [0, 1] per node (ghost probability).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GhostDetectorGAT(torch.nn.Module):
    """
    Graph Attention Network for binary ghost classification.

    Architecture:
        Input [N, F] → GATConv(heads=4) → ELU → Dropout
                     → GATConv(heads=1) → Sigmoid → [N]

    Parameters
    ----------
    in_channels      : Number of input node features
    hidden_channels  : Hidden dim per attention head (default 32)
    heads            : Number of attention heads in layer 1 (default 4)
    dropout          : Dropout probability (default 0.3)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
        )
        # Second layer collapses the heads into a single scalar
        self.conv2 = GATConv(
            hidden_channels * heads,
            1,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x               : Node feature matrix [N, F]
        edge_index      : Edge connectivity [2, E]
        return_attention: If True, return (logits, attention_weights)
                          where attention_weights is from conv1.

        Returns
        -------
        Ghost probabilities [N] in [0, 1], or tuple if return_attention=True.
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        if return_attention:
            x, (edge_idx, alpha) = self.conv1(x, edge_index, return_attention_weights=True)
        else:
            x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        out = torch.sigmoid(x).squeeze(-1)

        if return_attention:
            return out, (edge_idx, alpha)
        return out


class GhostDetectorGCN(torch.nn.Module):
    """
    Graph Convolutional Network baseline for ghost classification.

    Architecture:
        Input [N, F] → GCNConv → ReLU → Dropout
                     → GCNConv → Sigmoid → [N]

    Parameters
    ----------
    in_channels     : Number of input node features
    hidden_channels : Hidden dimension (default 32)
    dropout         : Dropout probability (default 0.3)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.conv2 = GCNConv(hidden_channels, 1, add_self_loops=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        Ghost probabilities [N] in [0, 1].
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x).squeeze(-1)


# ── Training utilities ────────────────────────────────────────────────────────

def train_one_epoch(
    model: torch.nn.Module,
    data,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
) -> float:
    """Run one training epoch, return loss value."""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask].float())
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Evaluate model on a given mask.

    Returns dict with: loss, accuracy, true_positives, false_positives,
                       false_negatives, true_negatives.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    model.eval()
    criterion = torch.nn.BCELoss()
    out = model(data.x, data.edge_index)

    probs = out[mask].cpu().numpy()
    true = data.y[mask].cpu().numpy()
    pred = (probs >= threshold).astype(int)

    loss = float(criterion(out[mask], data.y[mask].float()))
    acc = float((pred == true).mean())

    # Guard: sklearn metrics need at least 2 classes in test set
    has_both_classes = len(set(true)) == 2
    if has_both_classes:
        prec = precision_score(true, pred, zero_division=0)
        rec = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
    else:
        prec = rec = f1 = float("nan")

    return {
        "loss": loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "probs": probs,
        "true": true,
        "pred": pred,
    }
