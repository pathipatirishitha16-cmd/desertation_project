"""
cooccurrence.py
===============
Builds a 14×14 co-occurrence matrix from the NIH ChestX-ray14 labels,
runs statistical significance tests, and extracts association rules
using the Apriori algorithm.

Usage:
    python src/cooccurrence.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
import json

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
FIG_DIR   = BASE_DIR / "results" / "figures"
MET_DIR   = BASE_DIR / "results" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

# ── Disease Labels ─────────────────────────────────────────────────────────────
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax", "No Finding"
]

LABEL_COLS = [f"disease_{l}" for l in DISEASE_LABELS]

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build co-occurrence matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_cooccurrence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise joint occurrence counts for all 14 diseases.
    Returns a (14 × 14) DataFrame where entry [i,j] = number of images
    where both disease i and disease j appear.
    """
    print("[INFO] Building co-occurrence matrix ...")

    # Use only training split to avoid data leakage
    train_df  = df[df["split"] == "train"]
    label_mat = train_df[LABEL_COLS].values.astype(int)    # (N, 14)

    # Matrix multiply: (14, N) × (N, 14) = (14, 14)
    comat = label_mat.T @ label_mat                        # joint counts

    comat_df = pd.DataFrame(comat,
                             index=DISEASE_LABELS,
                             columns=DISEASE_LABELS)

    print(f"[INFO] Co-occurrence matrix shape: {comat_df.shape}")
    return comat_df


def build_conditional_probability_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    P(disease_j | disease_i) = co_count(i,j) / count(i)
    Returns (14 × 14) DataFrame.
    """
    print("[INFO] Computing conditional probability matrix ...")

    train_df  = df[df["split"] == "train"]
    label_mat = train_df[LABEL_COLS].values.astype(int)
    comat     = label_mat.T @ label_mat          # joint counts (14,14)
    diag      = np.diag(comat)                   # marginal counts

    with np.errstate(divide="ignore", invalid="ignore"):
        cond_prob = np.where(diag[:, None] > 0,
                             comat / diag[:, None],
                             0.0)

    # Make a writable copy before filling diagonal
    cond_arr = np.array(cond_prob, dtype=float)   # writable copy
    np.fill_diagonal(cond_arr, 0.0)               # zero out P(A|A)

    cond_df = pd.DataFrame(cond_arr,
                            index=DISEASE_LABELS,
                            columns=DISEASE_LABELS)
    return cond_df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Chi-square significance test with Bonferroni correction
# ─────────────────────────────────────────────────────────────────────────────

def chi_square_test(df: pd.DataFrame,
                    alpha: float = 0.001) -> pd.DataFrame:
    """
    For every disease pair (i, j) compute chi-square test of independence.
    Apply Bonferroni correction for multiple comparisons.
    Returns DataFrame with columns:
        disease_a, disease_b, chi2, p_value, p_corrected,
        significant, phi_coefficient
    """
    print("[INFO] Running chi-square tests with Bonferroni correction ...")

    train_df  = df[df["split"] == "train"]
    n         = len(train_df)
    results   = []
    n_pairs   = len(DISEASE_LABELS) * (len(DISEASE_LABELS) - 1) // 2

    for i, da in enumerate(DISEASE_LABELS):
        for j, db in enumerate(DISEASE_LABELS):
            if j <= i:
                continue

            col_a = f"disease_{da}"
            col_b = f"disease_{db}"

            # 2×2 contingency table
            a_pos = train_df[col_a].values
            b_pos = train_df[col_b].values

            both   = ((a_pos == 1) & (b_pos == 1)).sum()
            only_a = ((a_pos == 1) & (b_pos == 0)).sum()
            only_b = ((a_pos == 0) & (b_pos == 1)).sum()
            neither= ((a_pos == 0) & (b_pos == 0)).sum()

            contingency = np.array([[both, only_a],
                                     [only_b, neither]])

            try:
                chi2, p, _, _ = chi2_contingency(contingency,
                                                   correction=False)
            except ValueError:
                chi2, p = 0.0, 1.0

            # Bonferroni-corrected p-value
            p_corrected = min(p * n_pairs, 1.0)

            # Phi coefficient (effect size for 2×2 table)
            phi = np.sqrt(chi2 / n) if n > 0 else 0.0

            results.append({
                "disease_a":    da,
                "disease_b":    db,
                "count_both":   int(both),
                "chi2":         round(float(chi2), 4),
                "p_value":      float(p),
                "p_corrected":  float(p_corrected),
                "significant":  p_corrected < alpha,
                "phi_coeff":    round(float(phi), 4)
            })

    results_df = pd.DataFrame(results)
    sig_count  = results_df["significant"].sum()
    print(f"[INFO] Significant pairs (p_corrected < {alpha}): "
          f"{sig_count} / {n_pairs}")

    results_df.to_csv(MET_DIR / "chi_square_results.csv", index=False)
    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Association rule mining (Apriori)
# ─────────────────────────────────────────────────────────────────────────────

def run_apriori(df:          pd.DataFrame,
                min_support: float = 0.005,
                min_confidence: float = 0.60) -> pd.DataFrame:
    """
    Run Apriori algorithm on multi-label disease data.
    min_support = 0.005 means at least 0.5% of positive images (≈420 images).
    Returns association rules DataFrame sorted by confidence descending.
    """
    print(f"[INFO] Running Apriori (min_support={min_support}, "
          f"min_confidence={min_confidence}) ...")

    train_df     = df[df["split"] == "train"]
    # Use only positive images (exclude 'No Finding' for rule mining)
    pos_df       = train_df[train_df["disease_No Finding"] == 0].copy()
    disease_cols = [c for c in LABEL_COLS if "No_Finding" not in c
                    and "No Finding" not in c]

    print(f"[INFO] Positive images for Apriori: {len(pos_df):,}")

    # Build boolean DataFrame required by mlxtend
    bool_df = pos_df[disease_cols].astype(bool)

    # Frequent itemsets
    frequent_items = apriori(bool_df,
                              min_support=min_support,
                              use_colnames=True)
    print(f"[INFO] Frequent itemsets found: {len(frequent_items):,}")

    if len(frequent_items) == 0:
        print("[WARN] No frequent itemsets. Try lowering min_support.")
        return pd.DataFrame()

    # Generate rules
    rules = association_rules(frequent_items,
                               metric="confidence",
                               min_threshold=min_confidence)

    rules = rules.sort_values("confidence", ascending=False).reset_index(drop=True)

    # Clean up column names for readability
    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda x: " + ".join(sorted(list(x)))
    )
    rules["consequents_str"] = rules["consequents"].apply(
        lambda x: " + ".join(sorted(list(x)))
    )

    # Save
    save_cols = ["antecedents_str", "consequents_str",
                 "support", "confidence", "lift",
                 "leverage", "conviction"]
    rules[save_cols].to_csv(MET_DIR / "association_rules.csv", index=False)

    print(f"[INFO] Association rules extracted: {len(rules):,}")
    print(f"\n[INFO] Top 10 rules by confidence:")
    print(rules[["antecedents_str", "consequents_str",
                  "confidence", "lift"]].head(10).to_string(index=False))

    return rules


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_cooccurrence_heatmap(comat_df: pd.DataFrame,
                               cond_df:  pd.DataFrame) -> None:
    """Save co-occurrence count and conditional probability heatmaps."""
    print("[INFO] Plotting co-occurrence heatmaps ...")

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    # Short labels for display
    short = [l.replace("_", "\n") for l in DISEASE_LABELS]

    # — Raw counts —
    sns.heatmap(comat_df,
                ax=axes[0],
                annot=True, fmt=".0f",
                xticklabels=short, yticklabels=short,
                cmap="YlOrRd",
                linewidths=0.5,
                annot_kws={"size": 7})
    axes[0].set_title("Disease Co-occurrence Counts\n(Training Set)",
                       fontsize=14, fontweight="bold")
    axes[0].tick_params(labelsize=8)

    # — Conditional probabilities —
    sns.heatmap(cond_df,
                ax=axes[1],
                annot=True, fmt=".2f",
                xticklabels=short, yticklabels=short,
                cmap="Blues",
                vmin=0, vmax=1,
                linewidths=0.5,
                annot_kws={"size": 7})
    axes[1].set_title("P(column | row)  –  Conditional Probability\n(Training Set)",
                       fontsize=14, fontweight="bold")
    axes[1].tick_params(labelsize=8)

    plt.suptitle("NIH ChestX-ray14: Disease Co-occurrence Analysis",
                  fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = FIG_DIR / "cooccurrence_heatmaps.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {out_path}")


def plot_disease_prevalence(df: pd.DataFrame) -> None:
    """Bar chart of disease prevalence across all splits."""
    print("[INFO] Plotting disease prevalence ...")

    counts = {l: df[f"disease_{l}"].sum() for l in DISEASE_LABELS}
    counts_s = pd.Series(counts).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(counts_s.index, counts_s.values,
                  color=plt.cm.tab20.colors[:len(DISEASE_LABELS)])
    ax.set_xlabel("Disease", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("Disease Prevalence in NIH ChestX-ray14",
                  fontsize=14, fontweight="bold")
    ax.set_xticklabels(counts_s.index, rotation=45, ha="right", fontsize=9)

    for bar, val in zip(bars, counts_s.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 200,
                f"{val:,}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = FIG_DIR / "disease_prevalence.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {out_path}")


def plot_top_rules(rules: pd.DataFrame, top_n: int = 20) -> None:
    """Horizontal bar chart of top association rules by confidence."""
    if rules.empty:
        print("[WARN] No rules to plot.")
        return

    top   = rules.head(top_n).copy()
    labels = top["antecedents_str"] + "  →  " + top["consequents_str"]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn(top["confidence"].values)
    bars = ax.barh(range(len(top)), top["confidence"].values,
                   color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_title(f"Top {top_n} Association Rules (by Confidence)",
                  fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()

    for bar, val in zip(bars, top["confidence"].values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    out_path = FIG_DIR / "top_association_rules.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Save co-occurrence data for knowledge graph
# ─────────────────────────────────────────────────────────────────────────────

def save_cooccurrence_data(comat_df:  pd.DataFrame,
                            cond_df:   pd.DataFrame,
                            chi_df:    pd.DataFrame,
                            rules_df:  pd.DataFrame) -> dict:
    """
    Save all co-occurrence data to processed/ folder.
    Returns edge list for knowledge graph construction.
    """
    comat_df.to_csv(PROC_DIR / "cooccurrence_counts.csv")
    cond_df.to_csv(PROC_DIR  / "cooccurrence_conditional.csv")

    # Build edge list: significant pairs with conditional prob > 10%
    edges = []
    for _, row in chi_df[chi_df["significant"]].iterrows():
        da, db = row["disease_a"], row["disease_b"]
        prob_ab = cond_df.loc[da, db]
        prob_ba = cond_df.loc[db, da]

        if prob_ab > 0.10:
            edges.append({
                "source":    da,
                "target":    db,
                "weight":    round(float(prob_ab), 4),
                "count":     int(row["count_both"]),
                "phi":       float(row["phi_coeff"])
            })
        if prob_ba > 0.10:
            edges.append({
                "source":    db,
                "target":    da,
                "weight":    round(float(prob_ba), 4),
                "count":     int(row["count_both"]),
                "phi":       float(row["phi_coeff"])
            })

    with open(PROC_DIR / "graph_edges.json", "w") as f:
        json.dump(edges, f, indent=2)

    print(f"[INFO] Graph edges saved: {len(edges)}")
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Co-occurrence Mining Pipeline")
    print("=" * 60)

    # Load processed metadata
    from preprocess import load_processed_df
    df = load_processed_df()

    # 1. Build matrices
    comat_df = build_cooccurrence_matrix(df)
    cond_df  = build_conditional_probability_matrix(df)

    # 2. Statistical tests
    chi_df   = chi_square_test(df, alpha=0.001)

    # 3. Association rules
    rules_df = run_apriori(df, min_support=0.005, min_confidence=0.60)

    # 4. Visualise
    plot_cooccurrence_heatmap(comat_df, cond_df)
    plot_disease_prevalence(df)
    if not rules_df.empty:
        plot_top_rules(rules_df)

    # 5. Save for knowledge graph
    edges = save_cooccurrence_data(comat_df, cond_df, chi_df, rules_df)

    print("\n[DONE] Co-occurrence mining complete.")