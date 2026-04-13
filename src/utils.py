"""
utils.py
========
Shared utility functions used across the project.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR  = BASE_DIR / "results" / "figures"
MET_DIR  = BASE_DIR / "results" / "metrics"

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Infiltration",
    "Mass", "Nodule", "Pleural_Thickening", "Pneumonia",
    "Pneumothorax", "No Finding"
]


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[INFO] Saved → {path}")


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def youden_threshold(fpr: np.ndarray,
                      tpr: np.ndarray,
                      thresholds: np.ndarray) -> float:
    """Find optimal classification threshold using Youden's J statistic."""
    J   = tpr - fpr
    idx = np.argmax(J)
    return float(thresholds[idx])


def plot_confusion_matrix(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            disease: str) -> None:
    """Plot and save confusion matrix for a single disease."""
    from sklearn.metrics import confusion_matrix
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {disease.replace('_',' ')}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=14, fontweight="bold")
    plt.colorbar(im)
    plt.tight_layout()
    out = FIG_DIR / f"cm_{disease}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()