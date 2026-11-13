"""Grouped bar plot: paper accuracy vs. reproduced accuracy (val1, val2) per model.

Reads ../repro_results.xlsx (sheet 'Ark1'). Missing values (results not in yet)
are skipped and marked with 'n/a' so the plot works on a partially filled sheet.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "repro_results.xlsx"
OUT = ROOT / "plot_figures" / "repro_barplot.png"

YMIN = 60  # lower y-limit; set to 0 for a full-height axis
ERR = "std"  # error bars: "std" (run-to-run spread) or "SE" (uncertainty of the mean)

# (column in sheet, error column or None, legend label, colour)
BARS = [
    ("Paper, val1", None, "Paper", "#9e9e9e"),
    ("Repro, val1", f"{ERR}, val1", f"Repro, val1 (± {ERR})", "#1f77b4"),
    ("Repro, val2", f"{ERR}, val2", f"Repro, val2 (± {ERR})", "#ff7f0e"),
]

# --- load: header is on the first row, first column is empty -----------------
df = pd.read_excel(XLSX, sheet_name="Ark1")
df = df.dropna(axis=1, how="all")
df = df.rename(columns={df.columns[0]: "Model"}).dropna(subset=["Model"])
df = df.set_index("Model")

models = df.index.tolist()
x = np.arange(len(models))
width = 0.8 / len(BARS)

fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(models)), 5))

for i, (col, se_col, label, color) in enumerate(BARS):
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
    se = (pd.to_numeric(df[se_col], errors="coerce").to_numpy()
          if se_col else np.full(len(models), np.nan))
    pos = x + (i - (len(BARS) - 1) / 2) * width
    has = ~np.isnan(vals)

    ax.bar(pos[has], vals[has], width, label=label, color=color)

    # error bars only where an SE value exists
    has_se = has & ~np.isnan(se)
    if has_se.any():
        ax.errorbar(pos[has_se], vals[has_se], yerr=se[has_se], fmt="none",
                    ecolor="black", elinewidth=1, capsize=4)

    for p, v, s, ok in zip(pos, vals, se, has):
        if ok:
            top = v + (0 if np.isnan(s) else s)
            ax.text(p, top + 1.5, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8, rotation=90)
        else:
            ax.text(p, YMIN + 1, "n/a", ha="center", va="bottom",
                    fontsize=8, color="grey", rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.set_ylim(YMIN, 100)
ax.set_xlim(-0.5, len(models) - 0.5)
ax.set_title("BCI IV-2a (A01): paper vs. reproduced accuracy", fontsize=14)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(BARS),
          frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved {OUT}")
plt.show()
