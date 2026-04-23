# =============================================================================
# MALL CUSTOMER SEGMENTATION — K-MEANS CLUSTERING
# Data Science & Machine Learning Project
# Tools: pandas, matplotlib, seaborn, scikit-learn
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings, os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG   = "#0F1117"
CARD_BG   = "#1A1D2E"
GRID_COL  = "#2A2D3E"
TEXT_COL  = "#E0E0E0"
ACCENT    = "#7C4DFF"

CLUSTER_PALETTE = ["#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF", "#FF6FC8"]

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "text.color":        TEXT_COL,
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.6,
    "axes.grid":         True,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
})

OUTPUT_DIR = "mall_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  ✔  Saved → {path}")
    plt.close(fig)

# =============================================================================
# STEP 1 — GENERATE / LOAD DATASET
# Tries 'mall_customers.csv' first; creates a realistic synthetic dataset if
# not found so the script runs out-of-the-box.
# =============================================================================

CSV_PATH = "mall_customers.csv"

SEGMENT_PROFILES = [
    # (n, income_mean, income_std, score_mean, score_std, age_mean, age_std)
    (80,  25, 8,  78, 10, 24, 4),   # Young, low income, high spender
    (90,  85, 12, 82, 8,  32, 6),   # High income, high spender  ← VIP
    (95,  55, 10, 50, 12, 38, 8),   # Mid income, mid spender    ← Typical
    (85,  25, 7,  22, 9,  45, 9),   # Low income, low spender
    (100, 82, 10, 18, 8,  42, 8),   # High income, low spender   ← Savers
]

def build_synthetic(seed=42):
    rng  = np.random.default_rng(seed)
    rows = []
    cid  = 1
    genders = ["Male", "Female"]
    for n, im, is_, sm, ss, am, as_ in SEGMENT_PROFILES:
        for _ in range(n):
            income = int(np.clip(rng.normal(im, is_), 10, 140))
            score  = int(np.clip(rng.normal(sm, ss),  1, 100))
            age    = int(np.clip(rng.normal(am, as_), 18, 70))
            rows.append({
                "CustomerID":    cid,
                "Gender":        rng.choice(genders),
                "Age":           age,
                "Annual Income (k$)":  income,
                "Spending Score (1-100)": score,
            })
            cid += 1
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

if os.path.exists(CSV_PATH):
    print(f"Loading '{CSV_PATH}' …")
    df = pd.read_csv(CSV_PATH)
else:
    print("'mall_customers.csv' not found → generating synthetic dataset …")
    df = build_synthetic()
    df.to_csv(CSV_PATH, index=False)
    print(f"Synthetic dataset saved as '{CSV_PATH}'")

print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")

# =============================================================================
# STEP 2 — INITIAL DATA EXPLORATION
# =============================================================================
print("=" * 60)
print("STEP 2 — INITIAL DATA EXPLORATION")
print("=" * 60)

print("\n── First 5 rows ──")
print(df.head().to_string())

print("\n── Data types ──")
print(df.dtypes)

print("\n── Basic statistics ──")
print(df.describe().round(2))

print("\n── Missing values ──")
mv = df.isnull().sum()
print(mv[mv > 0] if mv.any() else "No missing values ✔")

print("\n── Gender distribution ──")
print(df["Gender"].value_counts())

# =============================================================================
# STEP 3 — DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3 — DATA PREPROCESSING")
print("=" * 60)

# 3a. Handle missing values
for col in df.select_dtypes(include="number").columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)
        print(f"  Filled '{col}' with median")

# 3b. Select features for clustering
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[FEATURES].copy()

# 3c. Scale features
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\nFeatures selected  : {FEATURES}")
print(f"Scaling method     : StandardScaler (mean=0, std=1)")
print(f"Scaled data shape  : {X_scaled.shape}")

# =============================================================================
# STEP 4 — ELBOW METHOD + SILHOUETTE ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4 — OPTIMAL CLUSTER DETECTION")
print("=" * 60)

K_RANGE = range(2, 11)
inertias    = []
silhouettes = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

# ── Elbow + Silhouette side-by-side ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(DARK_BG)

ks = list(K_RANGE)

# Elbow curve
axes[0].plot(ks, inertias, "o-", color=ACCENT, linewidth=2.2,
             markersize=7, markerfacecolor="white")
axes[0].fill_between(ks, inertias, alpha=0.08, color=ACCENT)
axes[0].set_title("Elbow Method — Optimal K")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (Within-cluster SSE)")

# Silhouette scores
colors_sil = [CLUSTER_PALETTE[0] if s == max(silhouettes) else "#4D96FF"
              for s in silhouettes]
bars = axes[1].bar(ks, silhouettes, color=colors_sil, edgecolor=DARK_BG, linewidth=0.8)
best_k = ks[silhouettes.index(max(silhouettes))]
axes[1].bar(best_k, max(silhouettes), color="#FFD93D",
            edgecolor=DARK_BG, linewidth=0.8, label=f"Best K = {best_k}")
for bar, val in zip(bars, silhouettes):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.003, f"{val:.3f}",
                 ha="center", fontsize=8, color=TEXT_COL)
axes[1].set_title("Silhouette Score by K")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score (higher = better)")
axes[1].legend()

plt.suptitle("Choosing the Optimal Number of Clusters",
             fontsize=15, fontweight="bold", color="white", y=1.02)
plt.tight_layout()
save(fig, "01_elbow_silhouette.png")

OPTIMAL_K = best_k
print(f"\nOptimal K selected : {OPTIMAL_K}  (highest silhouette score = {max(silhouettes):.4f})")

# =============================================================================
# STEP 5 — TRAIN FINAL K-MEANS MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5 — TRAINING K-MEANS MODEL")
print("=" * 60)

kmeans = KMeans(n_clusters=OPTIMAL_K, init="k-means++", n_init=20, random_state=42)
kmeans.fit(X_scaled)

df["Cluster"] = kmeans.labels_
centroids_scaled = kmeans.cluster_centers_
centroids_orig   = scaler.inverse_transform(centroids_scaled)  # back to original scale

print(f"\nModel trained with K = {OPTIMAL_K}")
print(f"Inertia            : {kmeans.inertia_:.2f}")
print(f"Silhouette Score   : {silhouette_score(X_scaled, kmeans.labels_):.4f}")

print("\nCluster sizes:")
for c, count in df["Cluster"].value_counts().sort_index().items():
    print(f"  Cluster {c}: {count} customers ({count/len(df)*100:.1f}%)")

# =============================================================================
# STEP 6 — VISUALIZATION
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6 — VISUALIZATIONS")
print("=" * 60)

# ── 6a. Main Cluster Scatter Plot ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor(DARK_BG)

for cluster_id in range(OPTIMAL_K):
    mask = df["Cluster"] == cluster_id
    ax.scatter(
        df.loc[mask, "Annual Income (k$)"],
        df.loc[mask, "Spending Score (1-100)"],
        s=70, alpha=0.75,
        color=CLUSTER_PALETTE[cluster_id % len(CLUSTER_PALETTE)],
        edgecolors="white", linewidths=0.3,
        label=f"Cluster {cluster_id}",
        zorder=2,
    )

# Plot centroids
for i, (cx, cy) in enumerate(centroids_orig):
    ax.scatter(cx, cy, s=280, marker="*",
               color="white", edgecolors=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)],
               linewidths=1.8, zorder=5)
    ax.annotate(f"C{i}", (cx, cy),
                textcoords="offset points", xytext=(10, 6),
                fontsize=10, fontweight="bold",
                color=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)])

ax.set_title(f"Customer Segments — K-Means Clustering  (K={OPTIMAL_K})",
             pad=12, color="white")
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1–100)")
ax.legend(loc="upper left", framealpha=0.2)
plt.tight_layout()
save(fig, "02_cluster_scatter.png")

# ── 6b. Cluster Profile Heatmap ───────────────────────────────────────────────
profile_cols = ["Annual Income (k$)", "Spending Score (1-100)", "Age"]
cluster_means = df.groupby("Cluster")[profile_cols].mean().round(1)

fig, ax = plt.subplots(figsize=(9, 4))
fig.patch.set_facecolor(DARK_BG)
sns.heatmap(
    cluster_means.T, annot=True, fmt=".1f",
    cmap="RdYlGn", linewidths=0.5, linecolor=DARK_BG,
    ax=ax, cbar_kws={"shrink": 0.8},
)
ax.set_title("Cluster Profile Heatmap  (mean feature values per cluster)")
ax.set_xlabel("Cluster")
ax.set_ylabel("Feature")
plt.tight_layout()
save(fig, "03_cluster_heatmap.png")

# ── 6c. Income & Spending Box Plots ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(DARK_BG)

for ax, feature in zip(axes, FEATURES):
    data_by_cluster = [df.loc[df["Cluster"] == c, feature].values
                       for c in range(OPTIMAL_K)]
    bp = ax.boxplot(data_by_cluster, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color=TEXT_COL),
                    capprops=dict(color=TEXT_COL),
                    flierprops=dict(marker="o", color=TEXT_COL,
                                    markersize=4, alpha=0.5))
    for patch, color in zip(bp["boxes"], CLUSTER_PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_title(f"{feature} by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel(feature)
    ax.set_xticklabels([f"C{i}" for i in range(OPTIMAL_K)])

plt.suptitle("Feature Distribution per Cluster",
             fontsize=14, fontweight="bold", color="white", y=1.02)
plt.tight_layout()
save(fig, "04_boxplots.png")

# ── 6d. Gender Split per Cluster ─────────────────────────────────────────────
if "Gender" in df.columns:
    gender_counts = (
        df.groupby(["Cluster", "Gender"])
        .size().reset_index(name="Count")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)
    pivoted = gender_counts.pivot(index="Cluster", columns="Gender", values="Count").fillna(0)
    pivoted.plot(kind="bar", ax=ax,
                 color=["#4D96FF", "#FF6FC8"], edgecolor=DARK_BG,
                 width=0.6)
    ax.set_title("Gender Distribution by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Customer Count")
    ax.set_xticklabels([f"C{i}" for i in range(OPTIMAL_K)], rotation=0)
    ax.legend(title="Gender")
    plt.tight_layout()
    save(fig, "05_gender_by_cluster.png")

# ── 6e. Pairwise scatter + KDE (Age, Income, Score) ──────────────────────────
pair_df = df[["Age", "Annual Income (k$)", "Spending Score (1-100)", "Cluster"]].copy()
pair_df["Cluster"] = pair_df["Cluster"].astype(str)

fig = plt.figure(figsize=(12, 10))
fig.patch.set_facecolor(DARK_BG)
g = sns.pairplot(
    pair_df, hue="Cluster",
    palette={str(i): CLUSTER_PALETTE[i] for i in range(OPTIMAL_K)},
    plot_kws=dict(alpha=0.55, s=30, edgecolor="none"),
    diag_kind="kde",
)
g.figure.set_facecolor(DARK_BG)
for ax in g.axes.flatten():
    if ax:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_COL)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
g.figure.suptitle("Pairplot — Age, Income & Spending Score by Cluster",
                   y=1.01, fontsize=14, fontweight="bold", color="white")
save(g.figure, "06_pairplot.png")

# =============================================================================
# STEP 7 — CLUSTER ANALYSIS & SEGMENT NAMING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7 — CLUSTER ANALYSIS")
print("=" * 60)

# Auto-label each cluster based on income & spending score quartiles
def label_cluster(income, score):
    high_inc  = income  >= 60
    low_inc   = income  <  45
    high_sp   = score   >= 60
    low_sp    = score   <  40
    if high_inc and high_sp:  return "💎 VIP Spenders",       "High Income · High Spending"
    if high_inc and low_sp:   return "💰 Careful Savers",     "High Income · Low Spending"
    if low_inc  and high_sp:  return "🛍️  Impulsive Buyers",   "Low Income · High Spending"
    if low_inc  and low_sp:   return "💤 Budget Shoppers",    "Low Income · Low Spending"
    return                           "📊 Average Customers",  "Mid Income · Mid Spending"

cluster_stats = df.groupby("Cluster").agg(
    Count       =("Cluster",                "size"),
    Income_mean =("Annual Income (k$)",     "mean"),
    Score_mean  =("Spending Score (1-100)", "mean"),
    Age_mean    =("Age",                    "mean"),
).round(1).reset_index()

segment_names, segment_desc = [], []
for _, row in cluster_stats.iterrows():
    name, desc = label_cluster(row["Income_mean"], row["Score_mean"])
    segment_names.append(name)
    segment_desc.append(desc)

cluster_stats["Segment"]     = segment_names
cluster_stats["Description"] = segment_desc
df["Segment"] = df["Cluster"].map(dict(zip(cluster_stats["Cluster"], segment_names)))

print("\nCluster Summary:")
print(cluster_stats.to_string(index=False))

# ── 6f. Final Scatter with Segment Labels ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(DARK_BG)

for _, row in cluster_stats.iterrows():
    cid   = int(row["Cluster"])
    mask  = df["Cluster"] == cid
    ax.scatter(
        df.loc[mask, "Annual Income (k$)"],
        df.loc[mask, "Spending Score (1-100)"],
        s=65, alpha=0.70,
        color=CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)],
        edgecolors="white", linewidths=0.3, zorder=2,
    )

# Annotate centroids with segment name
for _, row in cluster_stats.iterrows():
    cid = int(row["Cluster"])
    cx, cy = centroids_orig[cid]
    ax.scatter(cx, cy, s=300, marker="*", zorder=5,
               color="white",
               edgecolors=CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)],
               linewidths=2)
    ax.annotate(
        row["Segment"],
        (cx, cy),
        textcoords="offset points", xytext=(12, 8),
        fontsize=9, fontweight="bold",
        color=CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)],
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor=CARD_BG, edgecolor=CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)],
                  alpha=0.85),
    )

ax.set_title("Customer Segmentation Map — Labeled Segments", pad=12)
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1–100)")

legend_patches = [
    mpatches.Patch(color=CLUSTER_PALETTE[int(r["Cluster"])],
                   label=f"C{int(r['Cluster'])}: {r['Segment']} ({int(r['Count'])} customers)")
    for _, r in cluster_stats.iterrows()
]
ax.legend(handles=legend_patches, loc="upper left",
          framealpha=0.25, fontsize=9)
plt.tight_layout()
save(fig, "07_labeled_segments.png")

# =============================================================================
# BUSINESS INSIGHTS REPORT
# =============================================================================
STRATEGIES = {
    "💎 VIP Spenders":      "Offer loyalty programs, exclusive previews, and premium memberships. "
                             "Upsell luxury products. Personal shoppers / concierge service.",
    "💰 Careful Savers":    "Target with investment/value offers. Highlight quality over price. "
                             "Savings bundles and long-term value messaging work best.",
    "🛍️  Impulsive Buyers":  "Flash sales, limited-time deals, and trendy/affordable products. "
                             "Social media campaigns and FOMO marketing are highly effective.",
    "💤 Budget Shoppers":   "Discount coupons, clearance sales, and BOGO deals. "
                             "Low-price positioning and essentials. Avoid premium upsells.",
    "📊 Average Customers": "Engagement rewards and mid-tier loyalty perks. "
                             "Personalized recommendations to nudge them toward higher spending.",
}

print("\n")
print("=" * 65)
print("     CUSTOMER SEGMENTATION — BUSINESS INSIGHTS REPORT")
print("=" * 65)

print(f"""
DATASET   : {len(df)} customers  |  Features used: Annual Income, Spending Score
MODEL     : K-Means (k-means++ init, {OPTIMAL_K} clusters)
SILHOUETTE: {silhouette_score(X_scaled, kmeans.labels_):.4f}  (closer to 1.0 = better separation)
""")

for _, row in cluster_stats.iterrows():
    name  = row["Segment"]
    cid   = int(row["Cluster"])
    strat = STRATEGIES.get(name, "Build deeper understanding before targeting.")
    print(f"  Cluster {cid} — {name}")
    print(f"  {'─' * 55}")
    print(f"  Customers : {int(row['Count'])}  ({int(row['Count'])/len(df)*100:.1f}% of total)")
    print(f"  Avg Income: ${row['Income_mean']:.0f}k   Avg Score: {row['Score_mean']:.0f}/100   Avg Age: {row['Age_mean']:.0f} yrs")
    print(f"  Profile   : {row['Description']}")
    print(f"  Strategy  : {strat}")
    print()

print("=" * 65)
print(f"All charts saved to ./{OUTPUT_DIR}/")
print("=" * 65)

from PIL import Image
img=Image.open("images/mall_segmentation_dashboard.png")
img.show()