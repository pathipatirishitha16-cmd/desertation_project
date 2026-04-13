"""
train_cnn.py
============
Complete training pipeline for ResNet-50 chest X-ray classifier.
Implements:
  - Weighted binary cross-entropy (handles class imbalance)
  - AdamW optimiser with ReduceLROnPlateau scheduler
  - Early stopping on validation AUROC
  - Apple MPS / CUDA / CPU device auto-detection
  - Saves best model weights to models/resnet50_best.pth

Usage:
    python src/train_cnn.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import (load_processed_df, get_dataloaders,
                         compute_class_weights, DISEASE_LABELS)
from cnn_model import build_resnet50

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_DIR  = BASE_DIR / "models"
MET_DIR    = BASE_DIR / "results" / "metrics"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE     = 16          # Reduced for Mac memory
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-4
NUM_EPOCHS     = 5
PATIENCE       = 10          # Early stopping patience
LR_PATIENCE    = 5           # ReduceLROnPlateau patience
LR_FACTOR      = 0.5
NUM_WORKERS    = 0           # MUST be 0 on Mac to prevent segfault


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Auto-detect best available device (MPS → CUDA → CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Using Apple MPS (Metal) GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU (training will be slow)")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────

class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss - MPS compatible version.
    Uses standard BCELoss with manual weighting to avoid MPS crashes.
    """
    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid first (model outputs raw logits)
        probs = torch.sigmoid(logits)
        # Clamp to avoid log(0)
        probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)
        # Weighted BCE manually
        weight = targets * self.pos_weight.to(logits.device) + (1 - targets)
        loss   = -(targets * torch.log(probs) +
                   (1 - targets) * torch.log(1 - probs))
        loss   = (loss * weight).mean()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# AUROC computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_auroc(all_labels: np.ndarray,
                   all_probs:  np.ndarray) -> dict:
    """
    Compute per-disease and mean AUROC.
    Skips diseases with only one class present in ground truth.
    """
    auroc_dict = {}
    valid_aucs = []

    for i, disease in enumerate(DISEASE_LABELS):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]

        if len(np.unique(y_true)) < 2:
            auroc_dict[disease] = None
            continue

        auc = roc_auc_score(y_true, y_score)
        auroc_dict[disease] = round(float(auc), 4)
        valid_aucs.append(auc)

    auroc_dict["mean"] = round(float(np.mean(valid_aucs)), 4) if valid_aucs else 0.0
    return auroc_dict


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model:     nn.Module,
                loader:    torch.utils.data.DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device:    torch.device,
                scaler:    GradScaler) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for imgs, labels, _ in tqdm(loader, desc="  Train", leave=False):
        imgs   = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # MPS and CPU do NOT support autocast — only CUDA does
        if device.type == "cuda":
            with autocast(device_type="cuda"):
                logits, _ = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Mac MPS or CPU — standard forward pass
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Validation epoch
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model:     nn.Module,
             loader:    torch.utils.data.DataLoader,
             criterion: nn.Module,
             device:    torch.device) -> tuple:
    """
    Run validation.
    Returns: (mean_loss, auroc_dict, all_labels, all_probs)
    """
    model.eval()
    total_loss  = 0.0
    n_batches   = 0
    all_labels  = []
    all_probs   = []

    for imgs, labels, _ in tqdm(loader, desc="  Val  ", leave=False):
        imgs   = imgs.to(device)
        labels = labels.to(device)

        logits, _ = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches  += 1

        probs = torch.sigmoid(logits) if not hasattr(logits, "sigmoid") \
                else logits           # model already applies sigmoid
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_probs  = np.vstack(all_probs)
    auroc      = compute_auroc(all_labels, all_probs)
    mean_loss  = total_loss / max(n_batches, 1)

    return mean_loss, auroc, all_labels, all_probs


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame) -> nn.Module:
    """Full training pipeline. Returns best model."""
    device = get_device()

    # Data
    loaders     = get_dataloaders(df, BATCH_SIZE, NUM_WORKERS)
    class_weights = compute_class_weights(df).to(device)

    # Model
    model = build_resnet50().to(device)

    # Loss, optimiser, scheduler
    criterion = WeightedBCELoss(class_weights)
    optimizer = optim.AdamW(model.parameters(),
                             lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=LR_PATIENCE, factor=LR_FACTOR
    )
    scaler = GradScaler() if device.type == "cuda" else None

    # Tracking
    best_auroc     = 0.0
    patience_count = 0
    history        = []

    print(f"\n{'='*60}")
    print("  Starting ResNet-50 Training")
    print(f"  Epochs: {NUM_EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"{'='*60}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_epoch(model, loaders["train"], criterion,
                                  optimizer, device, scaler)
        val_loss, val_auroc, _, _ = validate(model, loaders["val"],
                                              criterion, device)

        mean_auc = val_auroc["mean"]
        elapsed  = time.time() - t0

        print(f"Epoch [{epoch:03d}/{NUM_EPOCHS}] "
              f"TrainLoss: {train_loss:.4f}  "
              f"ValLoss: {val_loss:.4f}  "
              f"ValAUROC: {mean_auc:.4f}  "
              f"Time: {elapsed:.1f}s")

        # Log per-disease AUROC
        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss,   4),
            "val_auroc":  mean_auc,
            **{f"auc_{k}": v for k, v in val_auroc.items() if k != "mean"}
        })

        # Scheduler step
        scheduler.step(mean_auc)

        # Save best model
        if mean_auc > best_auroc:
            best_auroc     = mean_auc
            patience_count = 0
            save_path = MODEL_DIR / "resnet50_best.pth"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_auroc":   mean_auc,
                "val_auroc_per_disease": val_auroc,
            }, save_path)
            print(f"  ✓ New best AUROC: {best_auroc:.4f}  → saved to {save_path}")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{PATIENCE})")

        # Early stopping
        if patience_count >= PATIENCE:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch}.")
            break

    # Save training history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(MET_DIR / "cnn_training_history.csv", index=False)
    print(f"[INFO] Training history saved → {MET_DIR / 'cnn_training_history.csv'}")
    print(f"\n[DONE] Best Validation AUROC: {best_auroc:.4f}")

    # Load best weights
    ckpt  = torch.load(MODEL_DIR / "resnet50_best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Extract and save embeddings for GAT training
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model:  nn.Module,
                        loader: torch.utils.data.DataLoader,
                        device: torch.device,
                        split:  str = "train") -> None:
    """
    Run all images through the CNN and save 1024-dim embeddings to disk.
    These embeddings are used as node features in the GAT.
    """
    print(f"[INFO] Extracting CNN embeddings for split='{split}' ...")
    model.eval()

    all_embeddings = []
    all_labels     = []
    all_indices    = []

    for imgs, labels, names in tqdm(loader, desc=f"  Embed {split}"):
        imgs = imgs.to(device)
        _, emb = model(imgs)
        all_embeddings.append(emb.cpu())
        all_labels.append(labels.cpu())
        all_indices.extend(list(names))

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    labels_tensor     = torch.cat(all_labels,     dim=0)

    out_dir = BASE_DIR / "data" / "processed"
    torch.save(embeddings_tensor, out_dir / f"embeddings_{split}.pt")
    torch.save(labels_tensor,     out_dir / f"labels_{split}.pt")

    pd.DataFrame(all_indices, columns=["image_index"]).to_csv(
        out_dir / f"indices_{split}.csv", index=False
    )
    print(f"[INFO] Embeddings saved: {embeddings_tensor.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ResNet-50 Training Pipeline")
    print("=" * 60)

    df    = load_processed_df()
    model = train(df)

    # Extract embeddings for all splits (needed by GAT)
    device  = get_device()
    model   = model.to(device)
    loaders = get_dataloaders(df, batch_size=64, num_workers=NUM_WORKERS)

    for split in ["train", "val", "test"]:
        extract_embeddings(model, loaders[split], device, split)

    print("\n[DONE] CNN training and embedding extraction complete.")