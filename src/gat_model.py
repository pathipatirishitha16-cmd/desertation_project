"""
gat_model.py  (v2 — with attention weight extraction)
======================================================
3-layer Graph Attention Network for disease co-occurrence prediction.
NOW RETURNS attention weights so the Streamlit interface can visualise
which disease-to-disease edges the model attends to for each image.

Architecture:
  Input : CNN embedding (1024) || one-hot disease (15) = 1039-dim per node
  GAT-1 : GATConv(1039 → 512, 8 heads, concat=True)  → 4096  [attn saved]
  GAT-2 : GATConv(4096 → 256, 8 heads, concat=True)  → 2048  [attn saved]
  GAT-3 : GATConv(2048 →  64, 4 heads, concat=False) →   64  [attn saved]
  Pool  : Mean over 15 disease nodes                  → (B, 64)
  Out   : Linear(64 → 15) + Sigmoid                  → (B, 15)

Key change from v1:
  - forward() now accepts return_attention=True
  - When True, returns (probs, attention_dict) instead of just probs
  - attention_dict contains layer-wise averaged attention matrices (15×15)
    suitable for direct heatmap visualisation in Streamlit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Tuple, Dict, Optional

NUM_CLASSES   = 15
EMBEDDING_DIM = 1024
NODE_FEAT_DIM = EMBEDDING_DIM + NUM_CLASSES   # 1039


class GATCooccurrenceModel(nn.Module):
    """
    GAT for disease co-occurrence prediction with attention extraction.

    For each image:
      1.  CNN produces 1024-dim embedding.
      2.  Broadcast to all 15 disease nodes.
      3.  Concatenate with 15-dim one-hot disease identity → 1039-dim node.
      4.  3 GAT layers propagate through the disease knowledge graph.
          Each layer returns attention coefficients (edge-level weights).
      5.  Mean pool over 15 nodes → (B, 64).
      6.  Linear head → 15 co-occurrence probabilities.

    When return_attention=True in forward():
      Returns (probs, attn_dict) where attn_dict = {
          'layer1': Tensor (15, 15),  # averaged over heads, single-image
          'layer2': Tensor (15, 15),
          'layer3': Tensor (15, 15),
          'combined': Tensor (15, 15) # mean of all 3 layers
      }
    """

    def __init__(self,
                 in_dim:      int   = NODE_FEAT_DIM,
                 num_classes: int   = NUM_CLASSES,
                 dropout:     float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_p   = dropout

        # return_attention_weights=True makes GATConv return (out, attn)
        self.gat1 = GATConv(in_dim, 512,
                             heads=8, concat=True,
                             dropout=dropout,
                             add_self_loops=True)

        self.gat2 = GATConv(4096, 256,
                             heads=8, concat=True,
                             dropout=dropout,
                             add_self_loops=True)

        self.gat3 = GATConv(2048, 64,
                             heads=4, concat=False,
                             dropout=dropout,
                             add_self_loops=True)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
            nn.Sigmoid()
        )

    # ── Internal helper ────────────────────────────────────────────────────────

    def _build_inputs(self,
                      cnn_embeddings: torch.Tensor,
                      edge_index: torch.Tensor):
        """Build batched node features and edge_index from CNN embeddings."""
        B      = cnn_embeddings.size(0)
        device = cnn_embeddings.device
        C      = self.num_classes

        one_hot = torch.eye(C, device=device)
        img_exp = cnn_embeddings.unsqueeze(1).expand(-1, C, -1)
        oh_exp  = one_hot.unsqueeze(0).expand(B, -1, -1)
        x       = torch.cat([img_exp, oh_exp], dim=-1).reshape(B * C, NODE_FEAT_DIM)

        # Offset edge_index for each graph in the batch
        offsets = torch.arange(B, device=device).unsqueeze(1) * C
        ei      = edge_index.unsqueeze(0).expand(B, -1, -1)
        ei      = (ei + offsets.unsqueeze(2)).reshape(2, -1)

        return x, ei, B, C

    def _attn_to_matrix(self,
                         attn_tuple,
                         ei_batched: torch.Tensor,
                         B: int, C: int,
                         device) -> torch.Tensor:
        """
        Convert raw GAT attention output → averaged (C, C) matrix for image 0.

        attn_tuple : (edge_index, alpha)  where alpha shape (num_edges, heads)
        Returns     : (C, C) float tensor — row=source, col=target
        """
        ei_out, alpha = attn_tuple              # alpha: (E_total, heads)
        alpha_mean    = alpha.mean(dim=-1)      # (E_total,)  avg over heads

        # Take only edges belonging to the FIRST image in batch (offset 0)
        mask   = (ei_out[0] < C) & (ei_out[1] < C)
        src    = ei_out[0][mask]
        tgt    = ei_out[1][mask]
        weights= alpha_mean[mask].abs()         # attention is already softmaxed

        mat = torch.zeros(C, C, device=device)
        mat.index_put_((src, tgt), weights, accumulate=True)

        # Normalise rows
        row_sum = mat.sum(dim=1, keepdim=True).clamp(min=1e-8)
        mat     = mat / row_sum
        return mat.detach().cpu()

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self,
                cnn_embeddings: torch.Tensor,
                edge_index:     torch.Tensor,
                edge_attr:      torch.Tensor = None,
                batch:          torch.Tensor = None,
                return_attention: bool = False):
        """
        Args:
            cnn_embeddings   : (B, 1024)
            edge_index       : (2, E)
            edge_attr        : (E, 1)  optional edge weights
            batch            : ignored — handled internally
            return_attention : if True, also return attention matrices

        Returns:
            probs      : (B, 15) co-occurrence probabilities
            attn_dict  : dict of (15,15) attention matrices  [only if return_attention=True]
        """
        x, ei, B, C = self._build_inputs(cnn_embeddings, edge_index)
        device       = cnn_embeddings.device

        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # ── Layer 1 ───────────────────────────────────────────────────────────
        if return_attention:
            out1, attn1 = self.gat1(x, ei, return_attention_weights=True)
            x = F.elu(out1)
        else:
            x = F.elu(self.gat1(x, ei))

        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # ── Layer 2 ───────────────────────────────────────────────────────────
        if return_attention:
            out2, attn2 = self.gat2(x, ei, return_attention_weights=True)
            x = F.elu(out2)
        else:
            x = F.elu(self.gat2(x, ei))

        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # ── Layer 3 ───────────────────────────────────────────────────────────
        if return_attention:
            out3, attn3 = self.gat3(x, ei, return_attention_weights=True)
            x = F.elu(out3)
        else:
            x = F.elu(self.gat3(x, ei))

        # ── Pool + classify ───────────────────────────────────────────────────
        x     = x.reshape(B, C, 64).mean(dim=1)
        probs = self.classifier(x)

        if return_attention:
            m1  = self._attn_to_matrix(attn1, ei, B, C, device)
            m2  = self._attn_to_matrix(attn2, ei, B, C, device)
            m3  = self._attn_to_matrix(attn3, ei, B, C, device)
            combined = (m1 + m2 + m3) / 3.0
            attn_dict = {
                "layer1":   m1,
                "layer2":   m2,
                "layer3":   m3,
                "combined": combined
            }
            return probs, attn_dict

        return probs


def build_gat_model(dropout: float = 0.3) -> GATCooccurrenceModel:
    """Build and report GAT model."""
    model     = GATCooccurrenceModel(dropout=dropout)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] GAT Model | Total params    : {total:,}")
    print(f"[INFO] GAT Model | Trainable params: {trainable:,}")
    print(f"[INFO] GAT Model | Input dim       : {NODE_FEAT_DIM}")
    print(f"[INFO] GAT Model | Output classes  : {NUM_CLASSES}")
    return model