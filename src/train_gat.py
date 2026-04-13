"""
train_gat.py
============
Two-stage training pipeline for the Graph Attention Network:
  Stage 1: Freeze ResNet-50, train GAT only          (30 epochs)
  Stage 2: Fine-tune entire network end-to-end       (20 epochs)

Uses pre-extracted CNN embeddings from train_cnn.py.

Usage:
    python src/train_gat.py
"""

import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gat_model import build_gat_model, GATCooccurrenceModel, NUM_CLASSES
from cnn_model import build_resnet50
from preprocess import DISEASE_LABELS, load_processed_df, get_dataloaders
from train_cnn import get_device, WeightedBCELoss, compute_auroc

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MET_DIR   = BASE_DIR / "results" / "metrics"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
STAGE1_EPOCHS  = 30
STAGE2_EPOCHS  = 5
BATCH_SIZE     = 64
LR_GAT         = 1e-3
LR_FINETUNE    = 1e-5
WEIGHT_DECAY   = 1e-4
PATIENCE       = 8
NUM_WORKERS    = 4


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load pre-extracted embeddings (Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding_loaders(batch_size: int = BATCH_SIZE) -> dict:
    """
    Load pre-extracted CNN embeddings from disk.
    Returns DataLoaders of (embedding, label) pairs.
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        emb_path = PROC_DIR / f"embeddings_{split}.pt"
        lbl_path = PROC_DIR / f"labels_{split}.pt"

        if not emb_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {emb_path}\n"
                f"Please run train_cnn.py first to extract embeddings."
            )

        embeddings = torch.load(emb_path)
        labels     = torch.load(lbl_path)

        dataset = TensorDataset(embeddings, labels)
        shuffle = (split == "train")
        loaders[split] = DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=0)
        print(f"[INFO] Loaded {split:5s} embeddings: {embeddings.shape}")

    return loaders


def load_graph_tensors(device: torch.device) -> tuple:
    """Load edge_index and edge_attr tensors."""
    edge_index = torch.load(PROC_DIR / "edge_index.pt").to(device)
    edge_attr  = torch.load(PROC_DIR / "edge_attr.pt").to(device)
    print(f"[INFO] Graph: edge_index {edge_index.shape}, "
          f"edge_attr {edge_attr.shape}")
    return edge_index, edge_attr


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Training epoch (embedding-based, Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch_gat(gat:        GATCooccurrenceModel,
                     loader:     DataLoader,
                     edge_index: torch.Tensor,
                     edge_attr:  torch.Tensor,
                     criterion:  nn.Module,
                     optimizer:  optim.Optimizer,
                     device:     torch.device) -> float:
    """One training epoch using pre-extracted embeddings."""
    gat.train()
    total_loss = 0.0
    n_batches  = 0

    for embeddings, labels in tqdm(loader, desc="  GAT Train", leave=False):
        embeddings = embeddings.to(device)
        labels     = labels.to(device)

        optimizer.zero_grad()

        logits = gat(embeddings, edge_index, edge_attr)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gat.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_gat(gat:        GATCooccurrenceModel,
                  loader:     DataLoader,
                  edge_index: torch.Tensor,
                  edge_attr:  torch.Tensor,
                  criterion:  nn.Module,
                  device:     torch.device) -> tuple:
    """Validation epoch. Returns (mean_loss, auroc_dict)."""
    gat.eval()
    total_loss = 0.0
    n_batches  = 0
    all_labels = []
    all_probs  = []

    for embeddings, labels in tqdm(loader, desc="  GAT Val  ", leave=False):
        embeddings = embeddings.to(device)
        labels     = labels.to(device)

        logits = gat(embeddings, edge_index, edge_attr)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        n_batches  += 1

        all_labels.append(labels.cpu().numpy())
        all_probs.append(logits.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_probs  = np.vstack(all_probs)
    auroc      = compute_auroc(all_labels, all_probs)
    mean_loss  = total_loss / max(n_batches, 1)

    return mean_loss, auroc


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Stage 1 – Train GAT only
# ─────────────────────────────────────────────────────────────────────────────

def stage1_train_gat(device: torch.device) -> GATCooccurrenceModel:
    """
    Stage 1: ResNet-50 weights FROZEN.
    Train GAT using pre-extracted 1024-dim embeddings.
    """
    print(f"\n{'='*60}")
    print("  STAGE 1: Training GAT (ResNet-50 frozen)")
    print(f"  Epochs: {STAGE1_EPOCHS}  |  LR: {LR_GAT}")
    print(f"{'='*60}")

    loaders              = load_embedding_loaders(BATCH_SIZE)
    edge_index, edge_attr = load_graph_tensors(device)

    gat       = build_gat_model().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(gat.parameters(),
                            lr=LR_GAT,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    best_auroc     = 0.0
    patience_count = 0
    history        = []

    for epoch in range(1, STAGE1_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_epoch_gat(gat, loaders["train"],
                                      edge_index, edge_attr,
                                      criterion, optimizer, device)
        val_loss, val_auroc = validate_gat(gat, loaders["val"],
                                            edge_index, edge_attr,
                                            criterion, device)
        mean_auc = val_auroc["mean"]
        elapsed  = time.time() - t0

        print(f"S1 Epoch [{epoch:02d}/{STAGE1_EPOCHS}]  "
              f"TrainLoss: {train_loss:.4f}  "
              f"ValLoss: {val_loss:.4f}  "
              f"ValAUROC: {mean_auc:.4f}  "
              f"({elapsed:.1f}s)")

        history.append({"stage": 1, "epoch": epoch,
                          "train_loss": round(train_loss, 4),
                          "val_loss": round(val_loss, 4),
                          "val_auroc": mean_auc})

        scheduler.step(mean_auc)

        if mean_auc > best_auroc:
            best_auroc     = mean_auc
            patience_count = 0
            torch.save(gat.state_dict(), MODEL_DIR / "gat_stage1_best.pth")
            print(f"  ✓ New best (Stage 1): {best_auroc:.4f}")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print("[INFO] Early stopping (Stage 1).")
                break

    # Load best
    gat.load_state_dict(torch.load(MODEL_DIR / "gat_stage1_best.pth",
                                    map_location=device))
    print(f"\n[INFO] Stage 1 best AUROC: {best_auroc:.4f}")
    return gat, history


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Stage 2 – End-to-end fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def stage2_finetune(cnn:        nn.Module,
                     gat:        GATCooccurrenceModel,
                     df:         pd.DataFrame,
                     device:     torch.device,
                     s1_history: list) -> tuple:
    """
    Stage 2: Fine-tune both ResNet-50 AND GAT together end-to-end.
    Uses raw images (not pre-extracted embeddings).
    """
    print(f"\n{'='*60}")
    print("  STAGE 2: End-to-end fine-tuning (CNN + GAT)")
    print(f"  Epochs: {STAGE2_EPOCHS}  |  LR: {LR_FINETUNE}")
    print(f"{'='*60}")

    edge_index, edge_attr = load_graph_tensors(device)
    loaders               = get_dataloaders(df, batch_size=16,
                                             num_workers=NUM_WORKERS)

    # Combine parameters
    all_params = list(cnn.parameters()) + list(gat.parameters())
    optimizer  = optim.Adam(all_params, lr=LR_FINETUNE,
                             weight_decay=WEIGHT_DECAY)
    criterion  = nn.BCELoss()
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STAGE2_EPOCHS
    )

    best_auroc     = 0.0
    patience_count = 0
    history        = s1_history.copy()

    for epoch in range(1, STAGE2_EPOCHS + 1):
        t0 = time.time()

        cnn.train()
        gat.train()
        total_loss = 0.0
        n_batches  = 0

        for imgs, labels, _ in tqdm(loaders["train"],
                                     desc=f"  S2 Train", leave=False):
            imgs   = imgs.to(device)
            labels = labels.to(device)
            B      = imgs.size(0)

            optimizer.zero_grad()

            _, embeddings = cnn(imgs)    # (B, 1024)
            logits = gat(embeddings, edge_index, edge_attr)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        train_loss = total_loss / max(n_batches, 1)

        # Validate (using pre-extracted embeddings for speed)
        emb_loaders          = load_embedding_loaders(batch_size=64)
        val_loss, val_auroc  = validate_gat(gat, emb_loaders["val"],
                                              edge_index, edge_attr,
                                              criterion, device)
        mean_auc = val_auroc["mean"]
        elapsed  = time.time() - t0

        print(f"S2 Epoch [{epoch:02d}/{STAGE2_EPOCHS}]  "
              f"TrainLoss: {train_loss:.4f}  "
              f"ValLoss: {val_loss:.4f}  "
              f"ValAUROC: {mean_auc:.4f}  "
              f"({elapsed:.1f}s)")

        history.append({"stage": 2, "epoch": epoch,
                          "train_loss": round(train_loss, 4),
                          "val_loss": round(val_loss, 4),
                          "val_auroc": mean_auc})

        scheduler.step()

        if mean_auc > best_auroc:
            best_auroc     = mean_auc
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "cnn_state":   cnn.state_dict(),
                "gat_state":   gat.state_dict(),
                "val_auroc":   mean_auc,
                "val_auroc_per_disease": val_auroc,
            }, MODEL_DIR / "gat_best.pth")
            print(f"  ✓ New best (Stage 2): {best_auroc:.4f}")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print("[INFO] Early stopping (Stage 2).")
                break

    print(f"\n[INFO] Stage 2 best AUROC: {best_auroc:.4f}")

    # Save training history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(MET_DIR / "gat_training_history.csv", index=False)
    print(f"[INFO] History saved → {MET_DIR / 'gat_training_history.csv'}")

    # Load best weights
    ckpt = torch.load(MODEL_DIR / "gat_best.pth", map_location=device)
    cnn.load_state_dict(ckpt["cnn_state"])
    gat.load_state_dict(ckpt["gat_state"])

    return cnn, gat, history


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  GAT Training Pipeline (2-Stage)")
    print("=" * 60)

    device = get_device()
    df     = load_processed_df()

    # Stage 1
    gat, s1_history = stage1_train_gat(device)

    # Load CNN for Stage 2
    print("\n[INFO] Loading best CNN weights for Stage 2 ...")
    cnn  = build_resnet50().to(device)
    ckpt = torch.load(MODEL_DIR / "resnet50_best.pth", map_location=device)
    cnn.load_state_dict(ckpt["model_state"])

    # Stage 2
    cnn, gat, full_history = stage2_finetune(cnn, gat, df, device, s1_history)

    print("\n[DONE] GAT training complete.")