# =============================================================================
#  IRIS DATASET — DATA VISUALIZATION PIPELINE
#  Professional charts using pandas, matplotlib & seaborn
#  Ready to run • Beginner-friendly • Well-commented
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL STYLE SETUP
# =============================================================================
# A clean, consistent palette for the three Iris species.
# Using colorblind-friendly colours so every reader can distinguish them.
PALETTE   = {"setosa": "#4C72B0", "versicolor": "#55A868", "virginica": "#C44E52"}
FONT_BASE = "DejaVu Sans"

plt.rcParams.update({
    "figure.facecolor": "#F7F9FC",
    "axes.facecolor":   "#FFFFFF",
    "axes.edgecolor":   "#CCCCCC",
    "axes.linewidth":   0.8,
    "axes.grid":        True,
    "grid.color":       "#E5E5E5",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "font.family":      FONT_BASE,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
})

# =============================================================================
# STEP 1 — LOAD THE IRIS DATASET
# =============================================================================
print("=" * 60)
print("STEP 1: Loading the Iris Dataset")
print("=" * 60)

iris_raw = load_iris(as_frame=True)          # Returns a Bunch object with .frame
df       = iris_raw.frame.copy()

# Map numeric target (0,1,2) to human-readable species names
df["species"] = df["target"].map(
    {0: "setosa", 1: "versicolor", 2: "virginica"}
)
df.drop(columns="target", inplace=True)

# Nicer column names (drop units, use underscores)
df.rename(columns={
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)":  "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)":  "petal_width",
}, inplace=True)

print("✔  Dataset loaded successfully.\n")


# =============================================================================
# STEP 2 — QUICK DATASET OVERVIEW
# =============================================================================
print("=" * 60)
print("STEP 2: Dataset Overview")
print("=" * 60)

print(f"\n▶ Shape: {df.shape[0]} rows × {df.shape[1]} columns")

print("\n▶ First 5 rows:")
print(df.head().to_string(index=False))

print("\n▶ Data types & nulls:")
print(df.dtypes)
print(f"\n  Total missing values: {df.isnull().sum().sum()}")

print("\n▶ Descriptive statistics (numerical features):")
print(df.describe(include="number").round(2))

print("\n▶ Species counts:")
print(df["species"].value_counts().to_string())


# =============================================================================
# STEP 3 — VISUALIZATIONS
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: BAR CHART — Count of each species
# PATTERN: The dataset is perfectly balanced — 50 samples per species.
#          This matters for modelling (no class imbalance to correct).
# ─────────────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 5))
fig1.patch.set_facecolor("#F7F9FC")

species_counts = df["species"].value_counts().reindex(PALETTE.keys())
bars = ax1.bar(
    species_counts.index,
    species_counts.values,
    color=list(PALETTE.values()),
    edgecolor="white",
    linewidth=1.2,
    width=0.5,
    zorder=3,
)

# Annotate each bar with its count
for bar in bars:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.8,
        str(int(height)),
        ha="center", va="bottom",
        fontsize=11, fontweight="bold", color="#333333"
    )

ax1.set_title("Species Count in the Iris Dataset",
              fontsize=14, fontweight="bold", pad=14, color="#222222")
ax1.set_xlabel("Species",    fontsize=11, labelpad=8)
ax1.set_ylabel("Count",      fontsize=11, labelpad=8)
ax1.set_ylim(0, 62)
ax1.tick_params(axis="x", length=0)
ax1.spines[["top", "right"]].set_visible(False)

# Subtle colour band under each bar label
for i, (label, color) in enumerate(PALETTE.items()):
    ax1.get_xticklabels()           # force tick generation
    ax1.get_xaxis().get_major_ticks()[i].label1.set_color(color)
    ax1.get_xaxis().get_major_ticks()[i].label1.set_fontweight("bold")

plt.tight_layout()
fig1.savefig("images/chart1_bar.png", dpi=150, bbox_inches="tight")
print("\n✔  Chart 1 (Bar) saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: HISTOGRAM — Distribution of all four numerical features
# PATTERN: Petal length and width show clear bimodal distributions —
#          setosa clusters at the low end, while versicolor and virginica
#          overlap at higher values. Sepal measurements are more normally
#          distributed and harder to separate visually.
# ─────────────────────────────────────────────────────────────────────────────
features     = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
feat_labels  = ["Sepal Length (cm)", "Sepal Width (cm)",
                "Petal Length (cm)", "Petal Width (cm)"]

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
fig2.patch.set_facecolor("#F7F9FC")
fig2.suptitle("Feature Distributions by Species",
              fontsize=15, fontweight="bold", y=1.01, color="#222222")

for ax, feat, label in zip(axes2.flat, features, feat_labels):
    for species, color in PALETTE.items():
        subset = df[df["species"] == species][feat]
        ax.hist(subset, bins=14, alpha=0.65, color=color,
                edgecolor="white", linewidth=0.6, label=species, zorder=3)

    # Overlay a KDE curve for each species
    for species, color in PALETTE.items():
        subset = df[df["species"] == species][feat]
        subset.plot.kde(ax=ax, color=color, linewidth=2, zorder=4)

    ax.set_title(label, fontsize=11, fontweight="bold", color="#333333", pad=8)
    ax.set_xlabel("Value (cm)", fontsize=9)
    ax.set_ylabel("Frequency",  fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.7)

plt.tight_layout()
fig2.savefig("images/chart2_histogram.png", dpi=150, bbox_inches="tight")
print("✔  Chart 2 (Histogram) saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3: SCATTER PLOT — Sepal Length vs Petal Length
# PATTERN: Setosa forms a tight, well-separated cluster (small petals).
#          Versicolor and virginica overlap slightly, but virginica generally
#          has longer petals for the same sepal length — a useful separator
#          for classification models.
# ─────────────────────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 6))
fig3.patch.set_facecolor("#F7F9FC")

for species, color in PALETTE.items():
    subset = df[df["species"] == species]
    ax3.scatter(
        subset["sepal_length"], subset["petal_length"],
        c=color, label=species.capitalize(),
        s=65, alpha=0.80, edgecolors="white", linewidths=0.6, zorder=3
    )

# Add light regression lines per species to show the linear trend
for species, color in PALETTE.items():
    subset = df[df["species"] == species]
    sns.regplot(
        data=subset, x="sepal_length", y="petal_length",
        ax=ax3, scatter=False,
        line_kws={"color": color, "linewidth": 1.5, "linestyle": "--", "alpha": 0.6}
    )

ax3.set_title("Sepal Length vs Petal Length by Species",
              fontsize=14, fontweight="bold", pad=14, color="#222222")
ax3.set_xlabel("Sepal Length (cm)", fontsize=11, labelpad=8)
ax3.set_ylabel("Petal Length (cm)", fontsize=11, labelpad=8)
ax3.legend(title="Species", title_fontsize=10, fontsize=9,
           framealpha=0.85, edgecolor="#CCCCCC")
ax3.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig3.savefig("images/chart3_scatter.png", dpi=150, bbox_inches="tight")
print("✔  Chart 3 (Scatter) saved.")


# ─────────────────────────────────────────────────────────────────────────────
# BONUS CHART 4: CORRELATION HEATMAP
# PATTERN: Petal length and petal width are highly correlated (r ≈ 0.96),
#          which means they carry redundant information. Sepal width has a
#          weak (even negative) correlation with the petal measurements —
#          a good reason to explore it separately in models.
# ─────────────────────────────────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(7, 5.5))
fig4.patch.set_facecolor("#F7F9FC")

corr_matrix = df[features].corr()

sns.heatmap(
    corr_matrix,
    ax=ax4,
    annot=True,          # Show correlation values inside cells
    fmt=".2f",
    cmap="RdYlGn",       # Red = negative, Yellow = neutral, Green = positive
    vmin=-1, vmax=1,
    linewidths=0.5,
    linecolor="#EEEEEE",
    square=True,
    cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    annot_kws={"size": 10, "weight": "bold"},
    xticklabels=feat_labels,
    yticklabels=feat_labels,
)

ax4.set_title("Feature Correlation Heatmap",
              fontsize=14, fontweight="bold", pad=14, color="#222222")
ax4.tick_params(axis="x", rotation=20, labelsize=9)
ax4.tick_params(axis="y", rotation=0,  labelsize=9)

plt.tight_layout()
fig4.savefig("images/chart4_heatmap.png", dpi=150, bbox_inches="tight")
print("✔  Chart 4 (Heatmap) saved.")


# =============================================================================
# COMBINED SUMMARY FIGURE — all 4 charts in one A4-style layout
# =============================================================================
fig5 = plt.figure(figsize=(16, 12))
fig5.patch.set_facecolor("#F7F9FC")
gs   = gridspec.GridSpec(2, 2, figure=fig5, hspace=0.38, wspace=0.32)

ax_bar  = fig5.add_subplot(gs[0, 0])
ax_scat = fig5.add_subplot(gs[0, 1])
ax_heat = fig5.add_subplot(gs[1, 0])
ax_hist = fig5.add_subplot(gs[1, 1])

# ── BAR (re-draw) ─────────────────────────────────────────────────────────
bars2 = ax_bar.bar(
    species_counts.index, species_counts.values,
    color=list(PALETTE.values()), edgecolor="white", linewidth=1.2,
    width=0.5, zorder=3
)
for bar in bars2:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                str(int(h)), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#333333")
ax_bar.set_title("Species Count", fontsize=12, fontweight="bold", color="#222222")
ax_bar.set_xlabel("Species"); ax_bar.set_ylabel("Count")
ax_bar.set_ylim(0, 62)
ax_bar.spines[["top","right"]].set_visible(False)

# ── SCATTER (re-draw) ─────────────────────────────────────────────────────
for species, color in PALETTE.items():
    s = df[df["species"] == species]
    ax_scat.scatter(s["sepal_length"], s["petal_length"],
                    c=color, label=species.capitalize(),
                    s=50, alpha=0.80, edgecolors="white", linewidths=0.5, zorder=3)
    sns.regplot(data=s, x="sepal_length", y="petal_length", ax=ax_scat,
                scatter=False,
                line_kws={"color": color, "linewidth": 1.4,
                          "linestyle": "--", "alpha": 0.55})
ax_scat.set_title("Sepal Length vs Petal Length",
                  fontsize=12, fontweight="bold", color="#222222")
ax_scat.set_xlabel("Sepal Length (cm)"); ax_scat.set_ylabel("Petal Length (cm)")
ax_scat.legend(title="Species", fontsize=8, title_fontsize=9, framealpha=0.8)
ax_scat.spines[["top","right"]].set_visible(False)

# ── HEATMAP (re-draw) ─────────────────────────────────────────────────────
short_labels = ["Sep.Len", "Sep.Wid", "Pet.Len", "Pet.Wid"]
sns.heatmap(corr_matrix, ax=ax_heat, annot=True, fmt=".2f",
            cmap="RdYlGn", vmin=-1, vmax=1,
            linewidths=0.5, linecolor="#EEEEEE", square=True,
            cbar_kws={"shrink": 0.75},
            annot_kws={"size": 9, "weight": "bold"},
            xticklabels=short_labels, yticklabels=short_labels)
ax_heat.set_title("Feature Correlations", fontsize=12, fontweight="bold", color="#222222")
ax_heat.tick_params(axis="x", rotation=15, labelsize=8)
ax_heat.tick_params(axis="y", rotation=0,  labelsize=8)

# ── PETAL LENGTH HISTOGRAM (re-draw, one feature for clarity) ─────────────
for species, color in PALETTE.items():
    subset = df[df["species"] == species]["petal_length"]
    ax_hist.hist(subset, bins=14, alpha=0.60, color=color,
                 edgecolor="white", linewidth=0.6, label=species, zorder=3)
    subset.plot.kde(ax=ax_hist, color=color, linewidth=2, zorder=4)
ax_hist.set_title("Petal Length Distribution", fontsize=12, fontweight="bold", color="#222222")
ax_hist.set_xlabel("Petal Length (cm)"); ax_hist.set_ylabel("Frequency")
ax_hist.legend(fontsize=8, framealpha=0.75)
ax_hist.spines[["top","right"]].set_visible(False)

fig5.suptitle("Iris Dataset — Visual Analysis Dashboard",
              fontsize=17, fontweight="bold", y=1.01, color="#1A1A2E")

plt.savefig(r"C:\Users\kaviy\Downloads.png",
            dpi=160, bbox_inches="tight", facecolor="#F7F9FC")
print("✔  Chart 5 (Dashboard) saved.\n")
print("✅  All visualizations complete!\n")


from PIL import Image
img=Image.open("images/chart5_dashboard.png")
img.show()

