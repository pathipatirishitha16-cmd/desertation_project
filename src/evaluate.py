"""
evaluate.py
===========
Complete evaluation pipeline:
  - Per-disease AUROC with 95% confidence intervals (bootstrap)
  - DeLong's test: GAT vs CNN baseline comparison
  - Subgroup analysis (age, sex, co-occurrence count)
  - ROC curve plots for dissertation
  - Confusion matrices and classification reports

Usage:
    python src/evaluate.py
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                               classification_report, f1_score,
                               precision_score, recall_score)
from pathlib import Path
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import DISEASE_LABELS, load_processed_df, get_dataloaders
from cnn_model import build_resnet50
from gat_model import build_gat_model, NUM_CLASSES
from train_cnn import get_device, compute_auroc

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
FIG_DIR   = BASE_DIR / "results" / "figures"
MET_DIR   = BASE_DIR / "results" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Bootstrap confidence intervals for AUROC
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_auroc(y_true:    np.ndarray,
                     y_score:   np.ndarray,
                     n_boot:    int   = 10000,
                     ci:        float = 0.95) -> tuple:
    """
    Compute AUROC with 95% CI via bootstrap.
    Returns: (auroc, lower, upper)
    """
    rng    = np.random.default_rng(seed=42)
    aucs   = []
    n      = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt  = y_true[idx]
        ys  = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))

    aucs    = np.array(aucs)
    alpha   = (1 - ci) / 2
    lower   = float(np.percentile(aucs, alpha * 100))
    upper   = float(np.percentile(aucs, (1 - alpha) * 100))
    auroc   = float(roc_auc_score(y_true, y_score))
    return auroc, lower, upper


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DeLong's test (compare two AUROCs)
# ─────────────────────────────────────────────────────────────────────────────

def delong_test(y_true:   np.ndarray,
                 y_score1: np.ndarray,
                 y_score2: np.ndarray) -> tuple:
    """
    DeLong's test for comparing two AUROC values.
    Returns: (z_statistic, p_value)

    Simplified implementation based on:
    DeLong ER, DeLong DM, Clarke-Pearson DL (1988).
    """
    def _compute_midrank(x):
        J     = np.argsort(x)
        Z     = x[J]
        N     = len(x)
        T     = np.zeros(N)
        i     = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2    = np.empty(N)
        T2[J] = T + 1
        return T2

    def _fastDeLong(y_true, y_score1, y_score2):
        pos_idx  = np.where(y_true == 1)[0]
        neg_idx  = np.where(y_true == 0)[0]
        m, n     = len(pos_idx), len(neg_idx)

        results  = []
        for score in [y_score1, y_score2]:
            r      = _compute_midrank(score)
            auc    = (r[pos_idx].mean() - (m + 1) / 2) / n
            v10    = (r[pos_idx] - (m + 1) / 2) / n
            v01    = 1 - (r[neg_idx] - (n + 1) / 2) / m
            results.append((auc, v10, v01, m, n))

        auc1, v10_1, v01_1, m, n = results[0]
        auc2, v10_2, v01_2, _, _ = results[1]

        cov_10 = np.cov(v10_1, v10_2)[0, 1]
        cov_01 = np.cov(v01_1, v01_2)[0, 1]

        var1   = np.var(v10_1) / n + np.var(v01_1) / m
        var2   = np.var(v10_2) / n + np.var(v01_2) / m
        cov    = cov_10 / n + cov_01 / m

        return auc1, auc2, var1, var2, cov

    try:
        auc1, auc2, var1, var2, cov = _fastDeLong(y_true, y_score1, y_score2)
        diff    = auc1 - auc2
        se      = np.sqrt(max(var1 + var2 - 2 * cov, 1e-10))
        z       = diff / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        return float(z), float(p_value)
    except Exception:
        return 0.0, 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Get predictions from saved models
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_cnn_predictions(device: torch.device) -> tuple:
    """Load CNN and get test set predictions."""
    print("[INFO] Getting CNN predictions on test set ...")

    emb_path = PROC_DIR / "embeddings_test.pt"
    lbl_path = PROC_DIR / "labels_test.pt"

    if not emb_path.exists():
        raise FileNotFoundError("Run train_cnn.py first to extract embeddings.")

    # CNN probabilities from raw model output
    # We use the saved test embeddings + classifier head
    cnn  = build_resnet50(pretrained=False).to(device)
    ckpt = torch.load(MODEL_DIR / "resnet50_best.pth", map_location=device)
    cnn.load_state_dict(ckpt["model_state"])
    cnn.eval()

    embeddings = torch.load(emb_path).to(device)
    labels     = torch.load(lbl_path)

    batch_size = 256
    all_probs  = []

    for i in range(0, len(embeddings), batch_size):
        batch  = embeddings[i:i+batch_size]
        # Forward only through classifier head
        logits = cnn.classifier(batch)
        all_probs.append(logits.cpu().numpy())

    all_probs  = np.vstack(all_probs)
    all_labels = labels.numpy()

    print(f"[INFO] CNN predictions: {all_probs.shape}")
    return all_labels, all_probs


@torch.no_grad()
def get_gat_predictions(device: torch.device):
    """Load GAT predictions. Returns None if GAT not trained yet."""
    print("[INFO] Getting GAT predictions on test set ...")

    ckpt_path = MODEL_DIR / "gat_best.pth"
    if not ckpt_path.exists():
        print("[WARN] gat_best.pth not found — run train_gat.py first.")
        print("[INFO] Continuing with CNN-only evaluation.")
        return None

    ckpt       = torch.load(ckpt_path, map_location=device)
    gat        = build_gat_model().to(device)
    gat.load_state_dict(ckpt["gat_state"])
    gat.eval()

    edge_index = torch.load(PROC_DIR / "edge_index.pt").to(device)
    edge_attr  = torch.load(PROC_DIR / "edge_attr.pt").to(device)
    embeddings = torch.load(PROC_DIR / "embeddings_test.pt").to(device)

    all_probs  = []
    batch_size = 64

    for i in range(0, len(embeddings), batch_size):
        emb    = embeddings[i:i+batch_size]
        logits = gat(emb, edge_index, edge_attr)
        all_probs.append(logits.cpu().numpy())

    return np.vstack(all_probs)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Full evaluation
# ─────────────────────────────────────────────────────────────────────────────

def full_evaluation(y_true:     np.ndarray,
                     y_cnn:      np.ndarray,
                     y_gat:      np.ndarray) -> pd.DataFrame:
    """
    Compute per-disease AUROC with 95% CI for both models.
    Run DeLong's test for each disease.
    Returns summary DataFrame.
    """
    print("[INFO] Running full evaluation ...")
    rows = []

    for i, disease in enumerate(DISEASE_LABELS):
        yt = y_true[:, i]

        if len(np.unique(yt)) < 2:
            print(f"  [SKIP] {disease}: only one class present")
            continue

        # CNN
        cnn_auc, cnn_lo, cnn_hi = bootstrap_auroc(yt, y_cnn[:, i])

        # GAT
        gat_auc, gat_lo, gat_hi = bootstrap_auroc(yt, y_gat[:, i])

        # DeLong
        z, p = delong_test(yt, y_gat[:, i], y_cnn[:, i])

        rows.append({
            "Disease":       disease,
            "CNN_AUROC":     round(cnn_auc, 4),
            "CNN_CI_Lower":  round(cnn_lo,  4),
            "CNN_CI_Upper":  round(cnn_hi,  4),
            "GAT_AUROC":     round(gat_auc, 4),
            "GAT_CI_Lower":  round(gat_lo,  4),
            "GAT_CI_Upper":  round(gat_hi,  4),
            "DeLong_Z":      round(z, 3),
            "DeLong_P":      round(p, 4),
            "Significant":   p < 0.05,
            "GAT_Better":    gat_auc > cnn_auc
        })

    results_df = pd.DataFrame(rows)

    # Summary row
    print(f"\n{'─'*70}")
    print(f"{'Disease':25s}  {'CNN AUROC':12s}  {'GAT AUROC':12s}  {'p-value':10s}")
    print(f"{'─'*70}")
    for _, row in results_df.iterrows():
        sig = "✓" if row["Significant"] else " "
        print(f"{row['Disease']:25s}  "
              f"{row['CNN_AUROC']:.4f} [{row['CNN_CI_Lower']:.3f}-{row['CNN_CI_Upper']:.3f}]  "
              f"{row['GAT_AUROC']:.4f} [{row['GAT_CI_Lower']:.3f}-{row['GAT_CI_Upper']:.3f}]  "
              f"p={row['DeLong_P']:.4f} {sig}")

    cnn_mean = results_df["CNN_AUROC"].mean()
    gat_mean = results_df["GAT_AUROC"].mean()
    print(f"{'─'*70}")
    print(f"{'MEAN':25s}  {cnn_mean:.4f}             {gat_mean:.4f}")
    print(f"{'─'*70}\n")

    results_df.to_csv(MET_DIR / "auroc_comparison.csv", index=False)
    print(f"[INFO] Results saved → {MET_DIR / 'auroc_comparison.csv'}")

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Subgroup analysis
# ─────────────────────────────────────────────────────────────────────────────

def subgroup_analysis(y_true:  np.ndarray,
                       y_gat:   np.ndarray,
                       df_test: pd.DataFrame) -> None:
    """Stratify AUROC by age group, sex, and co-occurrence count."""
    print("[INFO] Running subgroup analysis ...")

    df_test = df_test.reset_index(drop=True)
    results = []

    # Age groups
    df_test["age_group"] = pd.cut(
        df_test["Patient Age"].fillna(df_test["Patient Age"].median()),
        bins=[0, 40, 60, 200],
        labels=["<40", "40-60", ">60"]
    )

    # Co-occurrence bucket
    label_cols = [f"disease_{l}" for l in DISEASE_LABELS]
    df_test["n_diseases"] = df_test[label_cols].sum(axis=1)
    df_test["co_bucket"]  = df_test["n_diseases"].clip(1, 4).astype(str)
    df_test.loc[df_test["n_diseases"] >= 4, "co_bucket"] = "4+"

    for group_col in ["age_group", "Patient Sex", "co_bucket"]:
        for group_val in df_test[group_col].dropna().unique():
            mask = (df_test[group_col] == group_val).values
            if mask.sum() < 50:
                continue

            yt_g   = y_true[mask]
            yg_g   = y_gat[mask]
            aucs   = []

            for i in range(len(DISEASE_LABELS)):
                if len(np.unique(yt_g[:, i])) < 2:
                    continue
                aucs.append(roc_auc_score(yt_g[:, i], yg_g[:, i]))

            if aucs:
                results.append({
                    "Group":      group_col,
                    "Value":      str(group_val),
                    "N":          int(mask.sum()),
                    "Mean_AUROC": round(float(np.mean(aucs)), 4)
                })

    sub_df = pd.DataFrame(results)
    sub_df.to_csv(MET_DIR / "subgroup_analysis.csv", index=False)
    print("[INFO] Subgroup results:")
    print(sub_df.to_string(index=False))
    print(f"[INFO] Saved → {MET_DIR / 'subgroup_analysis.csv'}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(y_true: np.ndarray,
                     y_cnn:  np.ndarray,
                     y_gat:  np.ndarray) -> None:
    """Plot ROC curves for all 14 diseases (GAT vs CNN)."""
    print("[INFO] Plotting ROC curves ...")

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()

    for i, (disease, ax) in enumerate(zip(DISEASE_LABELS, axes)):
        yt = y_true[:, i]
        if len(np.unique(yt)) < 2:
            ax.set_visible(False)
            continue

        # GAT curve
        fpr_g, tpr_g, _ = roc_curve(yt, y_gat[:, i])
        auc_g = roc_auc_score(yt, y_gat[:, i])

        # CNN curve
        fpr_c, tpr_c, _ = roc_curve(yt, y_cnn[:, i])
        auc_c = roc_auc_score(yt, y_cnn[:, i])

        ax.plot(fpr_g, tpr_g, color="#e63946", lw=2,
                label=f"GAT ({auc_g:.3f})")
        ax.plot(fpr_c, tpr_c, color="#457b9d", lw=2, linestyle="--",
                label=f"CNN ({auc_c:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
        ax.set_title(disease.replace("_", " "), fontsize=9, fontweight="bold")
        ax.set_xlabel("FPR", fontsize=7)
        ax.set_ylabel("TPR", fontsize=7)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(len(DISEASE_LABELS), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("ROC Curves: GAT vs CNN Baseline (Test Set)",
                  fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = FIG_DIR / "roc_curves_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {out_path}")


def plot_auroc_comparison(results_df: pd.DataFrame) -> None:
    """Forest plot of AUROC with CI for both models."""
    print("[INFO] Plotting AUROC comparison ...")

    df      = results_df.sort_values("GAT_AUROC", ascending=True)
    y_pos   = range(len(df))

    fig, ax = plt.subplots(figsize=(10, 8))

    # CNN
    ax.barh([y - 0.2 for y in y_pos],
             df["CNN_AUROC"],
             height=0.35, color="#457b9d", alpha=0.8, label="CNN Baseline")
    ax.errorbar([y - 0.2 for y in y_pos],
                 df["CNN_AUROC"],
                 xerr=[df["CNN_AUROC"] - df["CNN_CI_Lower"],
                        df["CNN_CI_Upper"] - df["CNN_AUROC"]],
                 fmt="none", color="black", capsize=3, linewidth=1)

    # GAT
    ax.barh([y + 0.2 for y in y_pos],
             df["GAT_AUROC"],
             height=0.35, color="#e63946", alpha=0.8, label="GAT (Ours)")
    ax.errorbar([y + 0.2 for y in y_pos],
                 df["GAT_AUROC"],
                 xerr=[df["GAT_AUROC"] - df["GAT_CI_Lower"],
                        df["GAT_CI_Upper"] - df["GAT_AUROC"]],
                 fmt="none", color="black", capsize=3, linewidth=1)

    # Target line
    ax.axvline(x=0.82, color="green", linestyle="--", linewidth=1.5,
                label="Target AUROC (0.82)")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["Disease"].str.replace("_", " "), fontsize=10)
    ax.set_xlabel("AUROC (95% CI)", fontsize=12)
    ax.set_title("Per-Disease AUROC: GAT vs CNN Baseline\n"
                  "Error bars = 95% Bootstrap CI",
                  fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0.4, 1.0)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = FIG_DIR / "auroc_comparison_forestplot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {out_path}")


def plot_training_history() -> None:
    """Plot training curves for both CNN and GAT."""
    print("[INFO] Plotting training history ...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, fname, title in [
        (axes[0], "cnn_training_history.csv", "ResNet-50 Training"),
        (axes[1], "gat_training_history.csv", "GAT Training"),
    ]:
        path = MET_DIR / fname
        if not path.exists():
            ax.set_visible(False)
            continue

        hist = pd.read_csv(path)
        ax.plot(hist["epoch"], hist["train_loss"],
                label="Train Loss", color="#e63946")
        ax.plot(hist["epoch"], hist["val_loss"],
                label="Val Loss",   color="#457b9d")

        ax2 = ax.twinx()
        ax2.plot(hist["epoch"], hist["val_auroc"],
                  label="Val AUROC", color="#2a9d8f", linestyle="--")
        ax2.set_ylabel("AUROC", color="#2a9d8f")
        ax2.tick_params(axis="y", labelcolor="#2a9d8f")
        ax2.set_ylim(0.5, 1.0)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax2.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = FIG_DIR / "training_history.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Evaluation Pipeline")
    print("=" * 60)

    device = get_device()
    df     = load_processed_df()

    # Get predictions
    y_true, y_cnn = get_cnn_predictions(device)
    y_gat         = get_gat_predictions(device)

    if y_gat is None:
        # CNN-only evaluation (before GAT is trained)
        print("\n[INFO] Running CNN-only evaluation ...")
        print(f"{'─'*50}")
        print(f"{'Disease':25s}  {'CNN AUROC':12s}")
        print(f"{'─'*50}")
        for i, disease in enumerate(DISEASE_LABELS):
            yt = y_true[:, i]
            if len(np.unique(yt)) < 2:
                continue
            auc = roc_auc_score(yt, y_cnn[:, i])
            print(f"{disease:25s}  {auc:.4f}")
        valid = [roc_auc_score(y_true[:,i], y_cnn[:,i])
                 for i in range(len(DISEASE_LABELS))
                 if len(np.unique(y_true[:,i])) >= 2]
        print(f"{'─'*50}")
        print(f"{'MEAN':25s}  {np.mean(valid):.4f}")
        print(f"\n[INFO] Run train_gat.py to get full GAT vs CNN comparison.")
        plot_training_history()
    else:
        # Full GAT vs CNN evaluation
        results_df = full_evaluation(y_true, y_cnn, y_gat)
        test_df    = df[df["split"] == "test"].reset_index(drop=True)
        subgroup_analysis(y_true, y_gat, test_df)
        plot_roc_curves(y_true, y_cnn, y_gat)
        plot_auroc_comparison(results_df)
        plot_training_history()

    print("\n[DONE] Evaluation complete.")
    print(f"[INFO] Figures → {FIG_DIR}")
    print(f"[INFO] Metrics → {MET_DIR}")