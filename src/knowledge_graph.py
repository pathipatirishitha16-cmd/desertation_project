"""
knowledge_graph.py
==================
Builds a directed weighted knowledge graph of disease co-occurrences
using NetworkX, then exports it to PyTorch Geometric format for the
Graph Attention Network.

Usage:
    python src/knowledge_graph.py
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch_geometric.data import Data
from pathlib import Path
from typing import Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
FIG_DIR  = BASE_DIR / "results" / "figures"
MET_DIR  = BASE_DIR / "results" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Disease Labels ─────────────────────────────────────────────────────────────
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax", "No Finding"
]
LABEL2IDX = {l: i for i, l in enumerate(DISEASE_LABELS)}
NUM_NODES = len(DISEASE_LABELS)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build NetworkX directed graph
# ─────────────────────────────────────────────────────────────────────────────

def build_knowledge_graph(edges_path: Path = PROC_DIR / "graph_edges.json",
                           df_path:    Path = PROC_DIR / "metadata.csv"
                           ) -> nx.DiGraph:
    """
    Build a directed weighted knowledge graph.
    Nodes  = diseases (with prevalence attribute).
    Edges  = statistically significant co-occurrence relationships
             (conditional probability > 10%, Bonferroni-corrected p < 0.001).

    Returns: nx.DiGraph
    """
    print("[INFO] Building knowledge graph ...")

    # Load edge data
    with open(edges_path) as f:
        edges = json.load(f)

    # Load metadata to compute node prevalences
    df       = pd.read_csv(df_path)
    train_df = df[df["split"] == "train"]
    total    = len(train_df)

    G = nx.DiGraph()

    # Add nodes
    for label in DISEASE_LABELS:
        col        = f"disease_{label}"
        count      = int(train_df[col].sum()) if col in train_df.columns else 0
        prevalence = count / total if total > 0 else 0.0
        G.add_node(label,
                   idx        = LABEL2IDX[label],
                   count      = count,
                   prevalence = round(prevalence, 4))

    # Add edges
    for e in edges:
        G.add_edge(e["source"], e["target"],
                   weight = e["weight"],
                   count  = e["count"],
                   phi    = e["phi"])

    print(f"[INFO] Graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    # Print top edges by weight
    edge_data = sorted(G.edges(data=True),
                        key=lambda x: x[2]["weight"],
                        reverse=True)
    print("[INFO] Top 10 edges by conditional probability:")
    for src, tgt, attr in edge_data[:10]:
        print(f"       {src:25s} → {tgt:25s}  "
              f"P={attr['weight']:.3f}  n={attr['count']}")

    return G


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Visualise the graph
# ─────────────────────────────────────────────────────────────────────────────

def visualize_graph(G: nx.DiGraph,
                    min_weight: float = 0.20) -> None:
    """
    Draw the disease knowledge graph.
    Node size  ∝ prevalence.
    Edge width ∝ conditional probability.
    Only edges above min_weight are drawn for clarity.
    """
    print(f"[INFO] Visualising graph (min_weight={min_weight}) ...")

    # Filter edges
    filtered_edges = [(u, v, d) for u, v, d in G.edges(data=True)
                      if d["weight"] >= min_weight]
    if not filtered_edges:
        print("[WARN] No edges above threshold. Lowering to 0.10 ...")
        min_weight     = 0.10
        filtered_edges = [(u, v, d) for u, v, d in G.edges(data=True)
                          if d["weight"] >= min_weight]

    H = G.edge_subgraph([(u, v) for u, v, _ in filtered_edges]).copy()

    # Layout
    pos = nx.spring_layout(H, seed=42, k=2.5)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor("#0d1b2a")
    fig.patch.set_facecolor("#0d1b2a")

    # Node sizes and colors
    prevalences = [G.nodes[n].get("prevalence", 0.01) for n in H.nodes()]
    node_sizes  = [max(300, p * 15000) for p in prevalences]

    # Color by disease group
    respiratory = {"Atelectasis","Consolidation","Effusion","Emphysema",
                   "Infiltration","Pneumonia","Pneumothorax"}
    cardiac     = {"Cardiomegaly","Edema"}
    neoplastic  = {"Mass","Nodule","Fibrosis","Pleural_Thickening"}

    node_colors = []
    for n in H.nodes():
        if n in respiratory: node_colors.append("#e63946")
        elif n in cardiac:   node_colors.append("#457b9d")
        elif n in neoplastic:node_colors.append("#2a9d8f")
        else:                node_colors.append("#e9c46a")

    # Edges
    edge_weights = [d["weight"] for _, _, d in H.edges(data=True)]
    max_w = max(edge_weights) if edge_weights else 1.0

    nx.draw_networkx_edges(H, pos,
                            width     = [3 * w / max_w for w in edge_weights],
                            edge_color= edge_weights,
                            edge_cmap = plt.cm.YlOrRd,
                            alpha     = 0.7,
                            arrows    = True,
                            arrowsize = 20,
                            ax        = ax)

    nx.draw_networkx_nodes(H, pos,
                            node_size  = node_sizes,
                            node_color = node_colors,
                            alpha      = 0.9,
                            ax         = ax)

    # Labels
    nx.draw_networkx_labels(H, pos,
                             font_size  = 8,
                             font_color = "white",
                             font_weight= "bold",
                             ax         = ax)

    # Legend
    patches = [
        mpatches.Patch(color="#e63946", label="Respiratory"),
        mpatches.Patch(color="#457b9d", label="Cardiac"),
        mpatches.Patch(color="#2a9d8f", label="Neoplastic/Fibrotic"),
        mpatches.Patch(color="#e9c46a", label="Other"),
    ]
    ax.legend(handles=patches, loc="lower left",
              facecolor="#1a2a3a", labelcolor="white", fontsize=9)

    ax.set_title("Disease Co-occurrence Knowledge Graph\n"
                  f"(Edges: conditional probability ≥ {min_weight:.0%})",
                  color="white", fontsize=14, fontweight="bold", pad=15)
    ax.axis("off")

    out_path = FIG_DIR / "knowledge_graph.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Export graph to PyTorch Geometric format
# ─────────────────────────────────────────────────────────────────────────────

def export_graph_for_gat(G: nx.DiGraph) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert NetworkX graph to PyTorch Geometric edge_index and edge_attr.

    Returns:
        edge_index : LongTensor  shape (2, E)  — source / target node indices
        edge_attr  : FloatTensor shape (E, 1)  — edge weights (cond. prob.)
    """
    print("[INFO] Exporting graph for PyTorch Geometric ...")

    edge_index_list = []
    edge_attr_list  = []

    for src, tgt, data in G.edges(data=True):
        if src in LABEL2IDX and tgt in LABEL2IDX:
            edge_index_list.append([LABEL2IDX[src], LABEL2IDX[tgt]])
            edge_attr_list.append([data["weight"]])

    if not edge_index_list:
        print("[WARN] No edges found – using fully-connected fallback.")
        for i in range(NUM_NODES):
            for j in range(NUM_NODES):
                if i != j:
                    edge_index_list.append([i, j])
                    edge_attr_list.append([1.0 / (NUM_NODES - 1)])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr_list,  dtype=torch.float)

    print(f"[INFO] edge_index: {edge_index.shape}")
    print(f"[INFO] edge_attr : {edge_attr.shape}")

    # Save tensors
    torch.save(edge_index, PROC_DIR / "edge_index.pt")
    torch.save(edge_attr,  PROC_DIR / "edge_attr.pt")
    print(f"[INFO] Saved edge_index.pt and edge_attr.pt → {PROC_DIR}")

    return edge_index, edge_attr


def get_node_feature_matrix(G: nx.DiGraph) -> torch.Tensor:
    """
    Build a simple node feature matrix (NUM_NODES × 2):
        [prevalence, in_degree_normalised]
    Used as initial node features before CNN embeddings are added.
    """
    features = []
    for label in DISEASE_LABELS:
        prev   = G.nodes[label].get("prevalence", 0.0)
        in_deg = G.in_degree(label) / max(G.number_of_nodes() - 1, 1)
        features.append([prev, in_deg])

    node_feats = torch.tensor(features, dtype=torch.float)
    torch.save(node_feats, PROC_DIR / "node_features.pt")
    print(f"[INFO] Node features saved: {node_feats.shape}")
    return node_feats


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Graph statistics for dissertation
# ─────────────────────────────────────────────────────────────────────────────

def graph_statistics(G: nx.DiGraph) -> dict:
    """Compute and save graph statistics for the dissertation."""
    stats = {
        "num_nodes":       G.number_of_nodes(),
        "num_edges":       G.number_of_edges(),
        "density":         round(nx.density(G), 4),
        "avg_in_degree":   round(
            sum(d for _, d in G.in_degree()) / G.number_of_nodes(), 2),
        "avg_out_degree":  round(
            sum(d for _, d in G.out_degree()) / G.number_of_nodes(), 2),
        "is_weakly_connected": nx.is_weakly_connected(G),
        "num_weakly_connected_components": nx.number_weakly_connected_components(G),
    }

    # Most connected diseases
    in_deg_sorted  = sorted(G.in_degree(),  key=lambda x: x[1], reverse=True)
    out_deg_sorted = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)
    stats["top_in_degree"]  = [(n, d) for n, d in in_deg_sorted[:5]]
    stats["top_out_degree"] = [(n, d) for n, d in out_deg_sorted[:5]]

    print("\n[INFO] Graph Statistics:")
    for k, v in stats.items():
        print(f"       {k:40s}: {v}")

    with open(MET_DIR / "graph_statistics.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Knowledge Graph Construction Pipeline")
    print("=" * 60)

    # 1. Build graph
    G = build_knowledge_graph()

    # 2. Visualise
    visualize_graph(G, min_weight=0.20)

    # 3. Export for GAT
    edge_index, edge_attr = export_graph_for_gat(G)

    # 4. Node features
    node_feats = get_node_feature_matrix(G)

    # 5. Statistics
    stats = graph_statistics(G)

    print("\n[DONE] Knowledge graph construction complete.")