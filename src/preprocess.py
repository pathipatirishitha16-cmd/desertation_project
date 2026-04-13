"""
preprocess.py
=============
Loads NIH ChestX-ray14 dataset (v2020 CSV format), parses labels,
normalises images with CLAHE, and creates patient-level stratified splits.

CSV column names (actual v2020 format):
    Image Index, Finding Labels, Follow-up #, Patient ID,
    Patient Age, Patient Sex, View Position,
    OriginalImage[Width, Height], OriginalImagePixelSpacing[x, y]

Usage:
    python src/preprocess.py
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch
import json
from tqdm import tqdm
import multiprocessing
import platform

# ── MacOS segfault fix ─────────────────────────────────────────────────────
if platform.system() == "Darwin":
    if multiprocessing.get_start_method(allow_none=True) is None:
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
RAW_DIR   = DATA_DIR / "raw"
PROC_DIR  = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH  = DATA_DIR / "Data_Entry_2017_v2020.csv"

# ── Disease Labels (15 classes including Hernia from v2020) ───────────────
# These match EXACTLY what appears in the Finding Labels column of the CSV
# (except Pleural_Thickening which has a space in the CSV)
DISEASE_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
    "No Finding"
]

# Map from our internal label → what it looks like in the CSV pipe-string
CSV_LABEL_MAP = {
    "Pleural_Thickening": "Pleural Thickening",
    # All others are identical in CSV and our label list
}

NUM_CLASSES = len(DISEASE_LABELS)        # 15
LABEL2IDX   = {label: i for i, label in enumerate(DISEASE_LABELS)}

# ── Image Settings ─────────────────────────────────────────────────────────
IMG_SIZE   = 224
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)


# ──────────────────────────────────────────────────────────────────────────
# 1.  Load and parse the CSV
# ──────────────────────────────────────────────────────────────────────────

def load_labels(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """
    Load Data_Entry_2017_v2020.csv.
    Returns clean DataFrame with standardised column names.
    """
    print(f"[INFO] Loading labels from: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"\nCSV not found: {csv_path}"
            f"\nMake sure Data_Entry_2017_v2020.csv is inside: {DATA_DIR}"
        )

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]   # strip whitespace

    print(f"[INFO] Columns in CSV: {list(df.columns)}")
    print(f"[INFO] Total rows    : {len(df):,}")

    # ── Rename columns to standard names ──────────────────────────────────
    rename_map = {}

    # 'Patient Gender' → 'Patient Sex'  (older CSV version uses Gender)
    if "Patient Gender" in df.columns and "Patient Sex" not in df.columns:
        rename_map["Patient Gender"] = "Patient Sex"

    # Split bracket columns that come from v2020 CSV
    bracket_fixes = {
        "OriginalImage[Width":         "Image Width",
        "Height]":                     "Image Height",
        "OriginalImagePixelSpacing[x": "Pixel Spacing X",
        "y]":                          "Pixel Spacing Y",
    }
    for old, new in bracket_fixes.items():
        if old in df.columns:
            rename_map[old] = new

    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"[INFO] Renamed columns: {rename_map}")

    # ── Validate required columns ─────────────────────────────────────────
    required = ["Image Index", "Patient ID", "Finding Labels",
                "Patient Age", "Patient Sex"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    # ── Keep useful columns only ──────────────────────────────────────────
    keep = required.copy()
    for opt in ["View Position", "Follow-up #", "Image Width", "Image Height"]:
        if opt in df.columns:
            keep.append(opt)
    df = df[keep].copy()

    # ── Fix Patient Age ───────────────────────────────────────────────────
    df["Patient Age"] = (
        df["Patient Age"]
          .astype(str)
          .str.extract(r"(\d+)")[0]
          .astype(float)
    )
    df["Patient Age"] = df["Patient Age"].clip(0, 100)

    # ── Standardise Patient Sex ───────────────────────────────────────────
    df["Patient Sex"] = (
        df["Patient Sex"]
          .astype(str)
          .str.strip()
          .str.upper()
          .replace({"MALE": "M", "FEMALE": "F", "NAN": "U"})
          .fillna("U")
    )

    print(f"[INFO] Sex counts: {df['Patient Sex'].value_counts().to_dict()}")
    return df


def parse_disease_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse pipe-separated 'Finding Labels' into 15 binary disease columns.

    Example row:  'Cardiomegaly|Effusion'
    Creates:      disease_Cardiomegaly=1, disease_Effusion=1, rest=0
    """
    print("\n[INFO] Parsing disease labels ...")

    # Discover all labels that actually appear in this CSV
    all_csv_labels = set()
    df["Finding Labels"].dropna().apply(
        lambda x: [all_csv_labels.add(t.strip())
                   for t in str(x).split("|")]
    )
    print(f"[INFO] Unique labels in CSV: {sorted(all_csv_labels)}")

    # Create one binary column per disease
    for label in DISEASE_LABELS:
        csv_label = CSV_LABEL_MAP.get(label, label)  # e.g. Pleural Thickening
        df[f"disease_{label}"] = df["Finding Labels"].apply(
            lambda x: 1 if csv_label in [t.strip() for t in str(x).split("|")]
            else 0
        )

    label_cols         = [f"disease_{l}" for l in DISEASE_LABELS]
    df["num_diseases"] = df[label_cols].sum(axis=1)

    total   = len(df)
    no_find = (df["disease_No Finding"] == 1).sum()
    multi   = (df["num_diseases"] > 1).sum()

    print(f"[INFO] 'No Finding' images : {no_find:,}  ({no_find/total:.1%})")
    print(f"[INFO] Multi-label images  : {multi:,}  ({multi/total:.1%})")
    print(f"\n[INFO] Per-disease counts:")
    for label in DISEASE_LABELS:
        n   = df[f"disease_{label}"].sum()
        pct = n / total * 100
        print(f"       {label:25s}: {n:7,}  ({pct:.1f}%)")

    return df


# ──────────────────────────────────────────────────────────────────────────
# 2.  Find image paths on disk
# ──────────────────────────────────────────────────────────────────────────

def find_image_paths(df: pd.DataFrame,
                     raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Walk raw_dir recursively. Build {filename: full_path} map.
    Adds 'image_path' column. Drops rows with no matching file.
    """
    print(f"\n[INFO] Scanning: {raw_dir}")

    if not raw_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {raw_dir}")

    path_map = {}
    for root, _, files in os.walk(raw_dir):
        for fname in files:
            if fname.lower().endswith(".png"):
                path_map[fname] = os.path.join(root, fname)

    print(f"[INFO] PNG files found on disk: {len(path_map):,}")

    if len(path_map) == 0:
        raise FileNotFoundError(
            f"No PNG images found under {raw_dir}\n"
            "Please extract NIH tar archives there."
        )

    df = df.copy()
    df["image_path"] = df["Image Index"].map(path_map)

    missing = df["image_path"].isna().sum()
    matched = df["image_path"].notna().sum()
    if missing:
        print(f"[WARN] {missing:,} CSV entries have no matching image (dropped)")
    print(f"[INFO] Images matched to CSV: {matched:,}")

    df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────
# 3.  Image preprocessing helpers
# ──────────────────────────────────────────────────────────────────────────

def apply_clahe(img_gray: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement on a grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    return clahe.apply(img_gray)


def load_and_preprocess_image(image_path: str,
                               size: int = IMG_SIZE) -> np.ndarray:
    """
    Load PNG → grayscale → CLAHE → resize → RGB uint8 (size, size, 3).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot read image: {image_path}")
    img = apply_clahe(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


# ──────────────────────────────────────────────────────────────────────────
# 4.  Patient-level stratified split  70 / 15 / 15
# ──────────────────────────────────────────────────────────────────────────

def stratified_split(df:          pd.DataFrame,
                     train_ratio: float = 0.70,
                     val_ratio:   float = 0.15,
                     seed:        int   = 42) -> pd.DataFrame:
    """
    Split at PATIENT level so no patient appears in more than one split.
    Stratifies on: age quartile × sex × co-occurrence bucket.
    Adds 'split' column: 'train' | 'val' | 'test'
    """
    print("\n[INFO] Creating patient-level stratified splits (70/15/15) ...")

    patients = (
        df.groupby("Patient ID")
          .agg(age       = ("Patient Age",  "first"),
               sex       = ("Patient Sex",  "first"),
               n_diseases= ("num_diseases", "max"))
          .reset_index()
    )

    patients["age_q"] = pd.qcut(
        patients["age"].fillna(patients["age"].median()),
        q=4, labels=False, duplicates="drop"
    ).astype(str)

    patients["co_bucket"] = patients["n_diseases"].clip(0, 3).astype(str)

    patients["strat_key"] = (
        patients["age_q"] + "_" +
        patients["sex"].fillna("U") + "_" +
        patients["co_bucket"]
    )

    # Remove keys with < 3 members (can't stratify split)
    counts = patients["strat_key"].value_counts()
    good   = counts[counts >= 3].index
    patients = patients[patients["strat_key"].isin(good)].copy()

    # 70 / 30
    train_ids, temp_ids = train_test_split(
        patients["Patient ID"],
        test_size    = 1.0 - train_ratio,
        stratify     = patients["strat_key"],
        random_state = seed
    )

    # 15 / 15  from the 30%
    temp = patients[patients["Patient ID"].isin(temp_ids)].copy()
    counts2 = temp["strat_key"].value_counts()
    good2   = counts2[counts2 >= 2].index
    temp    = temp[temp["strat_key"].isin(good2)]

    val_ids, test_ids = train_test_split(
        temp["Patient ID"],
        test_size    = 0.50,
        stratify     = temp["strat_key"],
        random_state = seed
    )

    split_map = (
        {pid: "train" for pid in train_ids} |
        {pid: "val"   for pid in val_ids}   |
        {pid: "test"  for pid in test_ids}
    )

    df = df.copy()
    df["split"] = df["Patient ID"].map(split_map)
    df = df.dropna(subset=["split"]).reset_index(drop=True)

    for s in ["train", "val", "test"]:
        n = (df["split"] == s).sum()
        print(f"[INFO]   {s:5s}: {n:,} images")
    return df


# ──────────────────────────────────────────────────────────────────────────
# 5.  PyTorch Dataset & DataLoader
# ──────────────────────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for NIH ChestX-ray14 (v2020).
    Returns: (image_tensor [3,224,224], label_tensor [15], image_index str)
    """
    def __init__(self, df: pd.DataFrame, split: str, img_size: int = IMG_SIZE):
        self.df         = df[df["split"] == split].reset_index(drop=True)
        self.split      = split
        self.img_size   = img_size
        self.transform  = get_transforms(split)
        self.label_cols = [f"disease_{l}" for l in DISEASE_LABELS]
        print(f"[INFO] Dataset '{split}': {len(self.df):,} samples, "
              f"{NUM_CLASSES} classes")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = load_and_preprocess_image(row["image_path"], self.img_size)
        img   = Image.fromarray(img)
        img   = self.transform(img)
        label = torch.FloatTensor(row[self.label_cols].values.astype(float))
        return img, label, row["Image Index"]


def get_dataloaders(df:          pd.DataFrame,
                    batch_size:  int = 32,
                    num_workers: int = 0) -> dict:
    """
    Returns {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}.
    num_workers forced to 0 on Mac. pin_memory=False for MPS.
    """
    if platform.system() == "Darwin":
        num_workers = 0

    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = ChestXrayDataset(df, split)
        loaders[split] = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = False,
        )
    return loaders


# ──────────────────────────────────────────────────────────────────────────
# 6.  Save / Load helpers
# ──────────────────────────────────────────────────────────────────────────

def save_processed_df(df: pd.DataFrame,
                       path: Path = PROC_DIR / "metadata.csv") -> None:
    df.to_csv(path, index=False)
    print(f"[INFO] Saved → {path}  ({len(df):,} rows)")


def load_processed_df(path: Path = PROC_DIR / "metadata.csv") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"metadata.csv not found: {path}\n"
            "Run  python src/preprocess.py  first."
        )
    df = pd.read_csv(path)
    print(f"[INFO] Loaded metadata: {len(df):,} rows | "
          f"splits: {df['split'].value_counts().to_dict()}")
    return df


# ──────────────────────────────────────────────────────────────────────────
# 7.  Class weights
# ──────────────────────────────────────────────────────────────────────────

def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Inverse-frequency class weights → FloatTensor shape (15,).
    Classes with 0 positives get weight=1.0 (neutral, not penalised).
    Normalised so mean weight = 1.0.
    """
    train_df = df[df["split"] == "train"]
    n_train  = len(train_df)

    raw_weights = []
    print("\n[INFO] Class weights (pos count / neg count / weight):")
    for label in DISEASE_LABELS:
        col = f"disease_{label}"
        pos = int(train_df[col].sum())
        neg = n_train - pos

        if pos == 0:
            w = 1.0   # no positives → neutral weight
        else:
            w = float(neg) / float(pos)

        raw_weights.append(w)
        print(f"       {label:25s}: pos={pos:5,}  neg={neg:5,}  raw_w={w:.1f}")

    w_arr = np.array(raw_weights, dtype=np.float32)
    # Normalise so mean = 1.0  (keeps relative ratios, stabilises training)
    w_arr = w_arr / (w_arr.mean() + 1e-8)

    print(f"\n[INFO] Normalised weights:")
    for label, w in zip(DISEASE_LABELS, w_arr):
        print(f"       {label:25s}: {w:.3f}")

    return torch.FloatTensor(w_arr)


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  NIH ChestX-ray14 v2020  –  Preprocessing")
    print(f"  {NUM_CLASSES} classes: {DISEASE_LABELS}")
    print("=" * 60)

    df = load_labels(CSV_PATH)
    df = parse_disease_labels(df)
    df = find_image_paths(df, RAW_DIR)
    df = stratified_split(df)
    save_processed_df(df)
    weights = compute_class_weights(df)

    print("\n[INFO] Sanity-check: loading 1 batch ...")
    try:
        loaders = get_dataloaders(df, batch_size=4, num_workers=0)
        imgs, labels, names = next(iter(loaders["train"]))
        print(f"  ✓ Image  : {imgs.shape}")
        print(f"  ✓ Labels : {labels.shape}")
        print(f"  ✓ Files  : {list(names)}")
    except Exception as e:
        print(f"  [WARN] DataLoader test failed: {e}")
        print("  metadata.csv was saved. Continue to next step.")

    print("\n[DONE] Preprocessing complete.")
    print(f"       Next: python src/cooccurrence.py")