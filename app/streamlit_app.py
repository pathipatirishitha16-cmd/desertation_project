"""
streamlit_app.py  (v2 — with GAT Attention Visualisation)
==========================================================
Clinical web interface for Disease Co-occurrence GNN.

NEW in v2:
  ✦ GAT Attention Heatmap  — 15×15 matrix showing which disease pairs
    the model attends to for THIS specific image
  ✦ Attention Layer Selector — view Layer 1, Layer 2, Layer 3, or Combined
  ✦ Top Attention Edges panel — ranked list of strongest attention links
  ✦ Attention-weighted network graph — edge thickness = attention weight
  ✦ Explainability panel — maps attention scores to clinical interpretations
  ✦ Model confidence indicator with calibration warning

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
import cv2
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from PIL import Image
from pathlib import Path
import io

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from cnn_model  import build_resnet50, EMBEDDING_DIM
from gat_model  import build_gat_model, NUM_CLASSES
from preprocess import (DISEASE_LABELS, IMG_SIZE, apply_clahe,
                         get_transforms, LABEL2IDX)

MODEL_DIR = BASE_DIR / "models"
PROC_DIR  = BASE_DIR / "data" / "processed"
MET_DIR   = BASE_DIR / "results" / "metrics"

DISEASE_DISPLAY = [l.replace("_", " ") for l in DISEASE_LABELS]

# Colour coding by disease group (for graphs and charts)
DISEASE_GROUPS = {
    "Atelectasis":      ("Respiratory",      "#e63946"),
    "Consolidation":    ("Respiratory",      "#e63946"),
    "Effusion":         ("Respiratory",      "#e63946"),
    "Emphysema":        ("Respiratory",      "#e63946"),
    "Infiltration":     ("Respiratory",      "#e63946"),
    "Pneumonia":        ("Respiratory",      "#e63946"),
    "Pneumothorax":     ("Respiratory",      "#e63946"),
    "Cardiomegaly":     ("Cardiac",          "#457b9d"),
    "Edema":            ("Cardiac",          "#457b9d"),
    "Fibrosis":         ("Fibrotic/Tumour",  "#2a9d8f"),
    "Mass":             ("Fibrotic/Tumour",  "#2a9d8f"),
    "Nodule":           ("Fibrotic/Tumour",  "#2a9d8f"),
    "Pleural_Thickening":("Fibrotic/Tumour", "#2a9d8f"),
    "Hernia":           ("Other",            "#f4a261"),
    "No Finding":       ("Normal",           "#6c757d"),
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray Co-occurrence GNN",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main containers */
    .main-title {
        font-size: 2.0rem; font-weight: 900; color: #1a365d;
        text-align: center; padding: 0.8rem 0 0.2rem 0;
        border-bottom: 3px solid #2b6cb0; margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 0.9rem; color: #4a5568; text-align: center;
        margin-bottom: 1.5rem; font-style: italic;
    }
    /* Alert boxes */
    .alert-high {
        background: linear-gradient(135deg,#fff5f5,#fed7d7);
        border-left: 5px solid #e53e3e; padding: 0.8rem 1rem;
        border-radius: 8px; margin: 0.4rem 0; font-size: 0.9rem;
    }
    .alert-med {
        background: linear-gradient(135deg,#fffbeb,#fef3c7);
        border-left: 5px solid #d97706; padding: 0.8rem 1rem;
        border-radius: 8px; margin: 0.4rem 0; font-size: 0.9rem;
    }
    .info-box {
        background: linear-gradient(135deg,#ebf8ff,#bee3f8);
        border-left: 5px solid #3182ce; padding: 0.8rem 1rem;
        border-radius: 8px; margin: 0.4rem 0; font-size: 0.9rem;
    }
    /* Attention panel */
    .attn-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
    }
    .attn-score {
        font-size: 1.4rem; font-weight: 700; color: #2b6cb0;
    }
    /* Disclaimer */
    .disclaimer {
        background: #fff8f0; border: 1px solid #f6ad55;
        border-radius: 8px; padding: 0.8rem; font-size: 0.8rem;
        color: #744210; margin-top: 1rem;
    }
    /* Section headers */
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #2d3748;
        border-bottom: 2px solid #e2e8f0; padding-bottom: 0.3rem;
        margin: 1rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Model loading ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load CNN + GAT. Returns (cnn, gat, device, loaded_ok)."""
    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )

    cnn_path = MODEL_DIR / "resnet50_best.pth"
    gat_path = MODEL_DIR / "gat_best.pth"

    if not cnn_path.exists() or not gat_path.exists():
        return None, None, device, False

    cnn  = build_resnet50(pretrained=False).to(device)
    ckpt = torch.load(cnn_path, map_location=device)
    cnn.load_state_dict(ckpt["model_state"])
    cnn.eval()

    gat  = build_gat_model().to(device)
    ckpt = torch.load(gat_path, map_location=device)
    gat.load_state_dict(ckpt["gat_state"])
    gat.eval()

    return cnn, gat, device, True


@st.cache_resource
def load_graph_data():
    """Load edge tensors, conditional prob matrix, association rules."""
    edge_index = torch.load(PROC_DIR / "edge_index.pt")
    edge_attr  = torch.load(PROC_DIR / "edge_attr.pt")

    cond_path = PROC_DIR / "cooccurrence_conditional.csv"
    cond_df   = pd.read_csv(cond_path, index_col=0) if cond_path.exists() \
                else pd.DataFrame()

    rules_path = MET_DIR / "association_rules.csv"
    rules_df   = pd.read_csv(rules_path) if rules_path.exists() \
                 else pd.DataFrame()

    return edge_index, edge_attr, cond_df, rules_df


# ── Image preprocessing ────────────────────────────────────────────────────────

def preprocess_uploaded_image(uploaded_file) -> torch.Tensor:
    """Uploaded PNG/JPG → (1, 3, 224, 224) tensor."""
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img        = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Cannot decode image. Please upload PNG or JPG.")
    img = apply_clahe(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = Image.fromarray(img)
    tensor = get_transforms("test")(img).unsqueeze(0)
    return tensor


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(image_tensor, cnn, gat, device, edge_index, edge_attr):
    """
    Run CNN + GAT with attention extraction.

    Returns:
        cnn_probs  : np.ndarray (15,)
        gat_probs  : np.ndarray (15,)
        attn_dict  : dict of (15,15) tensors — per-layer attention matrices
        embedding  : np.ndarray (1024,)
    """
    image_tensor = image_tensor.to(device)
    ei           = edge_index.to(device)
    ea           = edge_attr.to(device)

    # CNN → probabilities + embedding
    cnn_probs_t, embedding = cnn(image_tensor)
    cnn_probs = cnn_probs_t.squeeze().cpu().numpy()
    embedding = embedding.cpu()

    # GAT → probabilities + attention weights
    gat_probs_t, attn_dict = gat(
        embedding.to(device), ei, ea,
        return_attention=True
    )
    gat_probs = gat_probs_t.squeeze().cpu().numpy()

    return cnn_probs, gat_probs, attn_dict, embedding.numpy().squeeze()


# ── Visualisation helpers ──────────────────────────────────────────────────────

def plot_attention_heatmap(attn_matrix: torch.Tensor,
                            layer_name:  str,
                            threshold:   float = 0.02) -> go.Figure:
    """
    Interactive Plotly heatmap of the (15×15) attention matrix.
    Rows = source disease, Cols = target disease.
    """
    mat = attn_matrix.numpy()
    # Zero out very low attention (noise)
    mat[mat < threshold] = 0.0

    labels = DISEASE_DISPLAY

    fig = px.imshow(
        mat,
        x=labels, y=labels,
        color_continuous_scale="Blues",
        zmin=0, zmax=mat.max(),
        aspect="auto",
        title=f"GAT Attention Weights — {layer_name}"
    )
    fig.update_layout(
        xaxis_title="Target Disease (attended to)",
        yaxis_title="Source Disease (attending from)",
        height=520,
        coloraxis_colorbar=dict(title="Attention<br>Weight"),
        title_font_size=14,
        xaxis=dict(tickangle=-45, tickfont_size=9),
        yaxis=dict(tickfont_size=9),
        margin=dict(l=130, r=40, t=60, b=140)
    )
    # Highlight diagonal (self-attention) differently
    for i in range(len(labels)):
        fig.add_shape(
            type="rect",
            x0=i - 0.5, x1=i + 0.5,
            y0=i - 0.5, y1=i + 0.5,
            line=dict(color="rgba(255,100,100,0.6)", width=1),
            fillcolor="rgba(0,0,0,0)"
        )
    return fig


def plot_attention_network(attn_matrix: torch.Tensor,
                            cnn_probs:   np.ndarray,
                            top_k:       int = 10) -> go.Figure:
    """
    Plotly network graph where edge thickness = attention weight.
    Node size = CNN detection probability.
    Only top-K edges shown for clarity.
    """
    mat = attn_matrix.numpy().copy()
    np.fill_diagonal(mat, 0)   # remove self-loops for display

    # Get top-K edges
    flat    = mat.flatten()
    top_idx = np.argsort(flat)[::-1][:top_k]
    rows    = top_idx // NUM_CLASSES
    cols    = top_idx %  NUM_CLASSES

    G = nx.DiGraph()
    for d in DISEASE_LABELS:
        G.add_node(d)
    for r, c in zip(rows, cols):
        w = float(mat[r, c])
        if w > 0:
            G.add_edge(DISEASE_LABELS[r], DISEASE_LABELS[c], weight=w)

    if len(G.edges()) == 0:
        return None

    pos = nx.spring_layout(G, seed=42, k=2.8)

    # Edge traces
    edge_traces = []
    for src, tgt, data in G.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        w       = data["weight"]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=max(1, w * 25), color=f"rgba(43,108,176,{min(w*5,0.9):.2f})"),
            hoverinfo="none", showlegend=False
        ))

    # Node traces
    node_x     = [pos[n][0] for n in G.nodes()]
    node_y     = [pos[n][1] for n in G.nodes()]
    node_text  = [n.replace("_", " ") for n in G.nodes()]
    node_sizes = [15 + float(cnn_probs[LABEL2IDX.get(n, 0)]) * 50
                  for n in G.nodes()]
    node_cols  = [float(cnn_probs[LABEL2IDX.get(n, 0)]) for n in G.nodes()]
    node_hover = [
        f"{n.replace('_',' ')}<br>"
        f"Detection: {float(cnn_probs[LABEL2IDX.get(n,0)]):.1%}<br>"
        f"In-degree attention: {sum(mat[:, DISEASE_LABELS.index(n)][:]):.3f}"
        for n in G.nodes()
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=8, color="#1a202c"),
        marker=dict(
            size=node_sizes,
            color=node_cols,
            colorscale="RdYlGn_r",
            cmin=0, cmax=1,
            colorbar=dict(title="Detection<br>Prob", thickness=10,
                          x=1.02, len=0.7),
            line=dict(width=1.5, color="white"),
            showscale=True
        ),
        hovertext=node_hover
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=dict(
                text="Attention-Weighted Disease Network<br>"
                     "<sup>Edge thickness = GAT attention | "
                     "Node size = detection probability</sup>",
                font=dict(size=13)
            ),
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="white",
            plot_bgcolor="#f8fafc",
            height=480,
            margin=dict(l=20, r=60, t=70, b=20)
        )
    )
    return fig


def plot_top_attention_edges(attn_matrix: torch.Tensor,
                              top_n: int = 8) -> go.Figure:
    """Horizontal bar chart of top attention-weighted disease pairs."""
    mat = attn_matrix.numpy().copy()
    np.fill_diagonal(mat, 0)

    flat      = mat.flatten()
    top_idx   = np.argsort(flat)[::-1][:top_n]
    rows      = top_idx // NUM_CLASSES
    cols      = top_idx %  NUM_CLASSES
    weights   = flat[top_idx]

    labels = [
        f"{DISEASE_DISPLAY[r]}  →  {DISEASE_DISPLAY[c]}"
        for r, c in zip(rows, cols)
        if weights[list(zip(rows, cols)).index((r, c))] > 0
    ]
    vals   = [w for w in weights if w > 0]

    if not labels:
        return None

    colors = [
        f"rgba(43,108,176,{min(0.3 + v*3, 0.95):.2f})"
        for v in vals
    ]

    fig = go.Figure(go.Bar(
        x=vals[::-1],
        y=labels[::-1],
        orientation="h",
        marker=dict(color=colors[::-1],
                    line=dict(color="white", width=0.5)),
        hovertemplate="%{y}<br>Attention: %{x:.4f}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="Top Attended Disease Relationships", font_size=13),
        xaxis_title="Attention Weight",
        height=320,
        margin=dict(l=200, r=30, t=50, b=40),
        plot_bgcolor="white",
        yaxis=dict(tickfont_size=9)
    )
    return fig


def render_prediction_bars(probs: np.ndarray,
                            title: str,
                            threshold: float) -> None:
    """Colour-coded prediction bars with percentage."""
    st.markdown(f'<div class="section-header">{title}</div>',
                unsafe_allow_html=True)
    sorted_idx = np.argsort(probs)[::-1]
    for idx in sorted_idx:
        disease = DISEASE_DISPLAY[idx]
        prob    = float(probs[idx])
        icon    = "🔴" if prob > threshold else ("🟡" if prob > 0.3 else "🟢")
        col1, col2 = st.columns([4, 1])
        with col1:
            st.progress(min(prob, 1.0), text=f"{icon} {disease}")
        with col2:
            st.write(f"**{prob:.1%}**")


def render_cooccurrence_alerts(cnn_probs, gat_probs, cond_df,
                                attn_combined, threshold) -> None:
    """Co-occurrence alerts enriched with attention scores."""
    st.markdown('<div class="section-header">⚠️ Co-occurrence Alerts</div>',
                unsafe_allow_html=True)

    detected = [DISEASE_LABELS[i] for i, p in enumerate(cnn_probs)
                if p > threshold and DISEASE_LABELS[i] != "No Finding"]

    if not detected:
        st.info("No primary diseases detected above threshold.")
        return

    attn_mat = attn_combined.numpy() if attn_combined is not None else None

    for primary in detected:
        pid = LABEL2IDX.get(primary, -1)
        st.markdown(f"**Detected: {primary.replace('_',' ')}** "
                    f"(CNN: {float(cnn_probs[pid]):.1%})")

        if not cond_df.empty and primary in cond_df.index:
            cond_probs = cond_df.loc[primary].drop(primary, errors="ignore")
            top_co     = cond_probs.sort_values(ascending=False).head(4)

            for secondary, epi_prob in top_co.items():
                if epi_prob < 0.10:
                    continue
                sid      = LABEL2IDX.get(secondary, -1)
                gat_conf = float(gat_probs[sid]) if sid >= 0 else epi_prob

                # Attention score for this pair
                attn_score = ""
                if attn_mat is not None and pid >= 0 and sid >= 0:
                    a = float(attn_mat[pid, sid])
                    attn_score = f" | 🧠 Attention: {a:.3f}"

                box_class = "alert-high" if gat_conf > threshold else "alert-med"
                st.markdown(
                    f'<div class="{box_class}">'
                    f'🔎 <b>{secondary.replace("_"," ")}</b> — '
                    f'Epidemiological: {epi_prob:.0%} | '
                    f'GAT confidence: {gat_conf:.1%}'
                    f'{attn_score}</div>',
                    unsafe_allow_html=True
                )
        st.divider()


def render_attention_explainability(attn_combined: torch.Tensor,
                                     cnn_probs: np.ndarray,
                                     threshold: float) -> None:
    """Plain-English explanation of what the attention shows."""
    st.markdown('<div class="section-header">🧠 Attention Explainability</div>',
                unsafe_allow_html=True)

    mat = attn_combined.numpy().copy()
    np.fill_diagonal(mat, 0)

    detected = [(DISEASE_LABELS[i], float(cnn_probs[i]))
                for i in range(NUM_CLASSES)
                if float(cnn_probs[i]) > threshold
                and DISEASE_LABELS[i] != "No Finding"]

    if not detected:
        st.info("No diseases detected — attention is broadly distributed.")
        return

    for disease, prob in detected[:4]:
        d_idx = LABEL2IDX.get(disease, -1)
        if d_idx < 0:
            continue

        # Which diseases does this node attend TO most?
        outgoing = mat[d_idx]
        top_out  = np.argsort(outgoing)[::-1][:3]

        # Which diseases attend TO this node most?
        incoming = mat[:, d_idx]
        top_in   = np.argsort(incoming)[::-1][:3]

        with st.expander(
            f"🔬 {disease.replace('_',' ')} — CNN: {prob:.1%}", expanded=True
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Model attends FROM this disease TO:**")
                for j in top_out:
                    if outgoing[j] > 0.01:
                        st.write(f"- {DISEASE_DISPLAY[j]}: "
                                 f"`{outgoing[j]:.3f}`")
            with col2:
                st.write("**Other diseases attending TO this one:**")
                for j in top_in:
                    if incoming[j] > 0.01:
                        st.write(f"- {DISEASE_DISPLAY[j]}: "
                                 f"`{incoming[j]:.3f}`")

            # Clinical interpretation
            out_names = [DISEASE_DISPLAY[j] for j in top_out
                         if outgoing[j] > 0.01]
            if out_names:
                st.success(
                    f"📋 **Clinical interpretation**: The GAT model, when "
                    f"processing {disease.replace('_',' ')}, directs most "
                    f"attention towards {', '.join(out_names[:2])}. This "
                    f"suggests the model has learned that these conditions "
                    f"frequently co-occur with {disease.replace('_',' ')} "
                    f"in the training data, consistent with epidemiological "
                    f"patterns in the NIH ChestX-ray14 dataset."
                )


# ── Main App ───────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown(
        '<div class="main-title">🫁 Chest X-Ray Disease Co-occurrence GNN</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">ResNet-50 + Graph Attention Network '
        '| NIH ChestX-ray14 (15 classes) '
        '| Disease Co-occurrence Prediction with Attention Visualisation</div>',
        unsafe_allow_html=True
    )

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        threshold   = st.slider("Detection Threshold", 0.10, 0.90, 0.50, 0.05,
                                 help="Diseases above this probability are flagged")
        attn_layer  = st.selectbox(
            "Attention Layer to Display",
            ["combined", "layer1", "layer2", "layer3"],
            index=0,
            help="Which GAT layer's attention weights to visualise"
        )
        top_k_edges = st.slider("Top-K Attention Edges (network graph)",
                                 3, 20, 10)
        attn_threshold = st.slider("Attention Display Threshold",
                                    0.0, 0.10, 0.02, 0.005,
                                    help="Hide attention weights below this value")

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        gat_path = MODEL_DIR / "gat_best.pth"
        if gat_path.exists():
            ckpt = torch.load(gat_path, map_location="cpu")
            auc  = ckpt.get("val_auroc", "N/A")
            st.metric("GAT Val AUROC", f"{auc:.4f}" if isinstance(auc, float) else auc)
            per = ckpt.get("val_auroc_per_disease", {})
            if per:
                best_d  = max(per, key=lambda k: per[k] or 0)
                st.metric("Best Disease", best_d.replace("_"," "),
                           f"{per[best_d]:.3f}")

        st.markdown("---")
        st.markdown("### 🎨 Colour Legend")
        for group, colour in [("Respiratory","#e63946"),
                               ("Cardiac","#457b9d"),
                               ("Fibrotic/Tumour","#2a9d8f"),
                               ("Other","#f4a261"),
                               ("Normal","#6c757d")]:
            st.markdown(
                f'<span style="background:{colour};color:white;'
                f'padding:2px 8px;border-radius:4px;font-size:0.8rem;">'
                f'{group}</span>', unsafe_allow_html=True
            )

    # ── Load models ────────────────────────────────────────────────────────────
    with st.spinner("Loading models..."):
        cnn, gat, device, loaded_ok = load_models()
        if not loaded_ok:
            st.error(
                "⚠️ Model weights not found. "
                "Please run training first:\n\n"
                "```\npython src\\train_cnn.py\npython src\\train_gat.py\n```"
            )
            st.stop()
        edge_index, edge_attr, cond_df, rules_df = load_graph_data()

    st.success(f"✅ Models loaded on **{device}**")

    # ── File uploader ──────────────────────────────────────────────────────────
    st.markdown("---")
    uploaded = st.file_uploader(
        "📤 Upload Chest X-Ray (PNG or JPG)",
        type=["png", "jpg", "jpeg"],
        help="Frontal-view (PA or AP) chest radiograph"
    )

    if uploaded is None:
        # Show dataset stats when no image uploaded
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.info("**Step 1** — Upload a chest X-ray image above")
        c2.info("**Step 2** — View primary disease predictions")
        c3.info("**Step 3** — Explore GAT attention visualisations")

        meta_path = PROC_DIR / "metadata.csv"
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            st.markdown("### 📊 Dataset Summary")
            a, b, c, d = st.columns(4)
            a.metric("Total Images", f"{len(meta):,}")
            b.metric("Train",  f"{(meta['split']=='train').sum():,}")
            c.metric("Val",    f"{(meta['split']=='val').sum():,}")
            d.metric("Test",   f"{(meta['split']=='test').sum():,}")
        return

    # ── Image display + inference ──────────────────────────────────────────────
    col_img, col_summary = st.columns([1, 2])

    with col_img:
        st.image(uploaded, caption="Uploaded X-Ray", use_column_width=True)

    with st.spinner("Running ResNet-50 + GAT inference..."):
        try:
            uploaded.seek(0)
            image_tensor = preprocess_uploaded_image(uploaded)
            cnn_probs, gat_probs, attn_dict, embedding = run_inference(
                image_tensor, cnn, gat, device, edge_index, edge_attr
            )
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.stop()

    attn_mat = attn_dict[attn_layer]   # (15, 15) tensor for selected layer

    # Summary stats
    detected = [DISEASE_LABELS[i] for i, p in enumerate(cnn_probs)
                if p > threshold and DISEASE_LABELS[i] != "No Finding"]
    gat_detected = [DISEASE_LABELS[i] for i, p in enumerate(gat_probs)
                    if p > threshold and DISEASE_LABELS[i] != "No Finding"]

    with col_summary:
        if detected:
            st.error(f"⚠️ **{len(detected)} patholog{'y' if len(detected)==1 else 'ies'}** "
                     f"detected (CNN > {threshold:.0%})")
        else:
            st.success(f"✅ No significant pathologies above {threshold:.0%} threshold")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CNN Detections", len(detected))
        m2.metric("GAT Detections", len(gat_detected))
        m3.metric("Highest CNN Prob",
                   f"{cnn_probs.max():.1%}",
                   DISEASE_DISPLAY[cnn_probs.argmax()])
        # Attention entropy (lower = more focused attention)
        mat_np = attn_mat.numpy().copy()
        np.fill_diagonal(mat_np, 0)
        attn_entropy = float(-np.sum(
            mat_np * np.log(mat_np + 1e-9)
        ) / (NUM_CLASSES * np.log(NUM_CLASSES)))
        m4.metric("Attention Focus",
                   "High" if attn_entropy < 0.3 else "Distributed",
                   f"entropy={attn_entropy:.2f}")

    # ── TABS ───────────────────────────────────────────────────────────────────
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Predictions",
        "🧠 Attention Heatmap",
        "🕸️ Attention Network",
        "📊 Top Attention Edges",
        "⚠️ Co-occurrence Alerts",
        "📋 Association Rules"
    ])

    # ── TAB 1: Predictions ────────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            render_prediction_bars(cnn_probs, "🖼️ CNN Primary Detection", threshold)
        with col_b:
            render_prediction_bars(gat_probs, "🕸️ GAT Co-occurrence Prediction", threshold)

        # Comparison chart
        st.markdown("---")
        st.markdown('<div class="section-header">CNN vs GAT Comparison</div>',
                    unsafe_allow_html=True)
        compare_df = pd.DataFrame({
            "Disease": DISEASE_DISPLAY,
            "CNN":     cnn_probs.tolist(),
            "GAT":     gat_probs.tolist()
        }).sort_values("GAT", ascending=False)

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name="CNN (Primary)", x=compare_df["Disease"],
            y=compare_df["CNN"], marker_color="#457b9d", opacity=0.8
        ))
        fig_compare.add_trace(go.Bar(
            name="GAT (Co-occurrence)", x=compare_df["Disease"],
            y=compare_df["GAT"], marker_color="#e63946", opacity=0.8
        ))
        fig_compare.add_hline(y=threshold, line_dash="dash",
                               line_color="green",
                               annotation_text=f"Threshold ({threshold:.0%})")
        fig_compare.update_layout(
            barmode="group", height=380,
            xaxis_tickangle=-45, xaxis_tickfont_size=9,
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=40, r=20, t=40, b=120),
            plot_bgcolor="white"
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    # ── TAB 2: Attention Heatmap ──────────────────────────────────────────────
    with tab2:
        st.markdown(
            f'<div class="info-box">Showing GAT attention weights for '
            f'<b>{attn_layer}</b>. Each cell [row → col] shows how much '
            f'the model attends from the <i>row disease</i> to the '
            f'<i>column disease</i> when making predictions for this image. '
            f'Brighter = stronger attention.</div>',
            unsafe_allow_html=True
        )

        fig_heatmap = plot_attention_heatmap(attn_mat, attn_layer,
                                              attn_threshold)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Show all 4 layers as small multiples
        st.markdown('<div class="section-header">All Layers Comparison</div>',
                    unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (lname, lmat) in enumerate(attn_dict.items()):
            with cols[i]:
                arr = lmat.numpy()
                np.fill_diagonal(arr, 0)
                fig_mini = px.imshow(
                    arr, color_continuous_scale="Blues",
                    aspect="auto", title=lname,
                    zmin=0, zmax=arr.max() if arr.max() > 0 else 1
                )
                fig_mini.update_layout(
                    height=240,
                    margin=dict(l=5, r=5, t=30, b=5),
                    coloraxis_showscale=False,
                    xaxis=dict(showticklabels=False),
                    yaxis=dict(showticklabels=False),
                    title_font_size=11
                )
                st.plotly_chart(fig_mini, use_container_width=True)

    # ── TAB 3: Attention Network ───────────────────────────────────────────────
    with tab3:
        st.markdown(
            '<div class="info-box">The network graph shows disease relationships '
            'weighted by GAT attention. <b>Thicker edges = stronger attention.</b> '
            'Node size scales with CNN detection probability. '
            'Only the top-K strongest attention edges are shown.</div>',
            unsafe_allow_html=True
        )
        fig_net = plot_attention_network(attn_mat, cnn_probs, top_k_edges)
        if fig_net:
            st.plotly_chart(fig_net, use_container_width=True)
        else:
            st.warning("No attention edges above threshold to display.")

        # Also show the original co-occurrence graph for comparison
        if not cond_df.empty:
            st.markdown('<div class="section-header">'
                        'Epidemiological Co-occurrence Graph (for comparison)'
                        '</div>', unsafe_allow_html=True)
            G2  = nx.DiGraph()
            for label in DISEASE_LABELS:
                G2.add_node(label)
            for i, src in enumerate(DISEASE_LABELS):
                for j, tgt in enumerate(DISEASE_LABELS):
                    if i != j and src in cond_df.index and tgt in cond_df.columns:
                        w = float(cond_df.loc[src, tgt])
                        if w > 0.20:
                            G2.add_edge(src, tgt, weight=w)

            pos2 = nx.spring_layout(G2, seed=42, k=2.8)
            e_traces = []
            for s, t, d in G2.edges(data=True):
                x0, y0 = pos2[s]; x1, y1 = pos2[t]
                e_traces.append(go.Scatter(
                    x=[x0,x1,None], y=[y0,y1,None], mode="lines",
                    line=dict(width=d["weight"]*6, color="rgba(200,50,50,0.5)"),
                    hoverinfo="none", showlegend=False
                ))
            n_x   = [pos2[n][0] for n in G2.nodes()]
            n_y   = [pos2[n][1] for n in G2.nodes()]
            n_txt = [n.replace("_"," ") for n in G2.nodes()]
            n_trace = go.Scatter(
                x=n_x, y=n_y, mode="markers+text",
                text=n_txt, textposition="top center",
                textfont=dict(size=8),
                marker=dict(size=16, color="#457b9d",
                            line=dict(width=1.5, color="white")),
                hoverinfo="text"
            )
            fig_epi = go.Figure(data=e_traces+[n_trace],
                layout=go.Layout(
                    title="Epidemiological Knowledge Graph (P > 0.20)",
                    height=420, showlegend=False,
                    xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    plot_bgcolor="#f8fafc",
                    margin=dict(l=20, r=20, t=50, b=20)
                ))
            st.plotly_chart(fig_epi, use_container_width=True)

    # ── TAB 4: Top Attention Edges ─────────────────────────────────────────────
    with tab4:
        st.markdown(
            '<div class="info-box">Ranked list of the strongest '
            'attention-weighted disease relationships for this specific image. '
            'These are the pairs the GAT model considers most important '
            'for its co-occurrence predictions.</div>',
            unsafe_allow_html=True
        )
        fig_bar = plot_top_attention_edges(attn_mat, top_n=12)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)

        # Table view
        st.markdown('<div class="section-header">Attention Matrix (Table View)</div>',
                    unsafe_allow_html=True)
        mat_np = attn_mat.numpy().copy()
        np.fill_diagonal(mat_np, 0)
        attn_df = pd.DataFrame(mat_np,
                                index=DISEASE_DISPLAY,
                                columns=DISEASE_DISPLAY)
        st.dataframe(
            attn_df.style.background_gradient(cmap="Blues", axis=None),
            height=300
        )

        # Explainability
        st.markdown("---")
        render_attention_explainability(attn_dict["combined"], cnn_probs, threshold)

    # ── TAB 5: Co-occurrence Alerts ────────────────────────────────────────────
    with tab5:
        render_cooccurrence_alerts(
            cnn_probs, gat_probs, cond_df, attn_dict["combined"], threshold
        )

    # ── TAB 6: Association Rules ───────────────────────────────────────────────
    with tab6:
        if rules_df.empty:
            st.warning("Association rules not found. Run cooccurrence.py first.")
        else:
            st.markdown(
                '<div class="info-box">Evidence-based association rules '
                'extracted from the NIH ChestX-ray14 training set using '
                'the Apriori algorithm (min support=0.5%, confidence≥60%).'
                '</div>', unsafe_allow_html=True
            )
            detected_names = [DISEASE_LABELS[i] for i, p in enumerate(cnn_probs)
                              if p > threshold]
            relevant = rules_df[
                rules_df["antecedents_str"].apply(
                    lambda x: any(d.replace("_"," ") in x or d in x
                                  for d in detected_names)
                )
            ].head(15)

            if relevant.empty:
                relevant = rules_df.head(10)
                st.info("No rules match detected diseases — showing top rules.")

            for _, row in relevant.iterrows():
                with st.expander(
                    f"**{row['antecedents_str']}**  →  "
                    f"**{row['consequents_str']}**  "
                    f"(conf: {row['confidence']:.0%})"
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Support",    f"{row['support']:.4f}")
                    c2.metric("Confidence", f"{row['confidence']:.1%}")
                    c3.metric("Lift",       f"{row['lift']:.2f}")
                    if row["lift"] > 1:
                        st.success(
                            f"This co-occurrence is {row['lift']:.1f}× more "
                            f"frequent than expected by chance."
                        )

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="disclaimer">'
        '⚕️ <b>Clinical Disclaimer</b>: This is a research prototype developed '
        'for academic purposes (7150CEM MSc Data Science, Coventry University). '
        'It must NOT be used for clinical diagnosis. Always consult a qualified '
        'radiologist for medical decisions. Model performance is validated on '
        'the NIH ChestX-ray14 dataset only.'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()