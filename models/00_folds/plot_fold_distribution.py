"""
Generate IEEE TNSRE-compliant 5-fold subject distribution chart
from Kfold5.json.

Usage:
    python models/00_folds/plot_fold_distribution.py
Output:
    models/00_folds/fold_distribution.pdf
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── IEEE TNSRE figure formatting ──────────────────────────────────
IEEE_COL_W  = 3.5
IEEE_DCOL_W = 7.16

mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":  "stix",
    "font.size":         14,
    "axes.titlesize":    14,
    "axes.labelsize":    14,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   12,
    "lines.linewidth":   1.5,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.02,
    "figure.dpi":        150,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

VAL_SUBJECT_FRAC = 0.18  # same as notebook

# ── Load data ─────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
with open(HERE / "Kfold5.json", "r", encoding="utf-8") as f:
    data = json.load(f)

folds = data["folds"]
n_folds = len(folds)

train_counts = []
val_counts   = []
test_counts  = []

for fold in folds:
    n_total_train = len(fold["train"])
    n_test  = len(fold["test"])
    n_val   = max(1, round(n_total_train * VAL_SUBJECT_FRAC))
    n_train = n_total_train - n_val
    train_counts.append(n_train)
    val_counts.append(n_val)
    test_counts.append(n_test)

train_counts = np.array(train_counts)
val_counts   = np.array(val_counts)
test_counts  = np.array(test_counts)

# ── Plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(IEEE_DCOL_W, IEEE_DCOL_W * 0.45))

y_pos = np.arange(n_folds)
labels = [f"Fold {i+1}" for i in range(n_folds)]

# Stacked horizontal bars — soft colors, light hatching
bars_train = ax.barh(y_pos, train_counts, height=0.6,
                     color="#A8CEE8", edgecolor="black", linewidth=0.8,
                     hatch="//", alpha=0.6, label="Train")

bars_val = ax.barh(y_pos, val_counts, height=0.6,
                   left=train_counts,
                   color="#FFDD77", edgecolor="black", linewidth=0.8,
                   hatch="\\\\", alpha=0.6, label="Validation")

bars_test = ax.barh(y_pos, test_counts, height=0.6,
                    left=train_counts + val_counts,
                    color="#C4E6A8", edgecolor="black", linewidth=0.8,
                    hatch="..", alpha=0.6, label="Test")

# Numbers inside each bar — black text
for bars, counts, offsets in [
    (bars_train, train_counts, np.zeros(n_folds)),
    (bars_val,   val_counts,   train_counts),
    (bars_test,  test_counts,  train_counts + val_counts),
]:
    for i, (bar, count, offset) in enumerate(zip(bars, counts, offsets)):
        x_center = offset + count / 2
        ax.text(x_center, y_pos[i], str(count),
                ha="center", va="center", fontsize=12,
                fontweight="bold", color="black")

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel("Number of Subjects")
ax.invert_yaxis()

fig.legend(*ax.get_legend_handles_labels(), frameon=True, edgecolor="black",
           ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.10, 1, 1])

out_path = HERE / "fold_distribution.pdf"
fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"Saved: {out_path}")
