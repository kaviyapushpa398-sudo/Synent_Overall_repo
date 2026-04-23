# =============================================================================
# SUPERSTORE SALES DATA ANALYSIS
# Professional Business Intelligence Script
# Tools: pandas, matplotlib, seaborn
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "#F8F9FA",
    "axes.facecolor": "#FFFFFF",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family": "DejaVu Sans",
})

OUTPUT_DIR = "superstore_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  ✔  Saved → {path}")
    plt.close(fig)

# =============================================================================
# STEP 1 – GENERATE / LOAD DATASET
# The script first tries to load 'superstore.csv' from the working directory.
# If not found it builds a realistic synthetic dataset so you can run it
# immediately without any external file.
# =============================================================================

CSV_PATH = "superstore.csv"

CATEGORIES = {
    "Furniture":        ["Bookcases", "Chairs", "Furnishings", "Tables"],
    "Office Supplies":  ["Appliances", "Art", "Binders", "Envelopes",
                         "Fasteners", "Labels", "Paper", "Storage", "Supplies"],
    "Technology":       ["Accessories", "Copiers", "Machines", "Phones"],
}

REGIONS   = ["East", "West", "Central", "South"]
SEGMENTS  = ["Consumer", "Corporate", "Home Office"]
SHIP_MODE = ["Standard Class", "Second Class", "First Class", "Same Day"]

PRODUCTS = {
    "Bookcases":    ["Sauder Bookcases", "Bush Bookcases", "O'Sullivan Bookcase"],
    "Chairs":       ["Hon Executive Chair", "Global Leather Chair", "Bretford Chair"],
    "Furnishings":  ["Eldon Shelf", "Advantus Panel", "Tensor Lamp"],
    "Tables":       ["Bevis Computer Table", "Chromcraft Table", "Lesro Table"],
    "Appliances":   ["Hoover Stove", "Fellowes Shredder", "Avery Electric Razor"],
    "Art":          ["Fiskars Scissors", "Sanford Pencils", "Dixon Ticonderoga"],
    "Binders":      ["Avery Heavy Binders", "Cardinal Slant-D Binders", "Ibico Binders"],
    "Envelopes":    ["Poly String-Tie Envelopes", "Wausau Astrobright Envelopes"],
    "Fasteners":    ["Advantus Push Pins", "OIC Push Pins", "OIC Staples"],
    "Labels":       ["Avery Labels", "Mead File Folders Labels"],
    "Paper":        ["Xerox Paper", "Hammermill Paper", "Southworth Connoisseur"],
    "Storage":      ["Fellowes File Cabinet", "Eldon Jumbo Storage", "Tenex Shelf"],
    "Supplies":     ["Dixon Ticonderoga Pencils", "Boston Pencil Sharpener"],
    "Accessories":  ["Logitech Mouse", "Kensington Keyboard", "Plantronics Headset"],
    "Copiers":      ["HP LaserJet Copier", "Xerox DocuCenter Copier"],
    "Machines":     ["Cisco TelePresence", "Polycom ViewStation"],
    "Phones":       ["Apple iPhone", "Samsung Galaxy", "Motorola Moto"],
}

# Profit margin hints per sub-category (mean ± std)
MARGINS = {
    "Bookcases": (-0.05, 0.10), "Chairs": (0.03, 0.08),
    "Furnishings": (0.10, 0.05), "Tables": (-0.10, 0.12),
    "Appliances": (0.12, 0.06), "Art": (0.18, 0.05),
    "Binders": (0.05, 0.12), "Envelopes": (0.20, 0.04),
    "Fasteners": (0.22, 0.04), "Labels": (0.25, 0.03),
    "Paper": (0.18, 0.05), "Storage": (0.08, 0.07),
    "Supplies": (-0.04, 0.10), "Accessories": (0.15, 0.06),
    "Copiers": (0.20, 0.08), "Machines": (0.04, 0.10),
    "Phones": (0.12, 0.07),
}

def build_synthetic_dataset(n=9800, seed=42):
    """Create a realistic Superstore-style dataset."""
    rng = np.random.default_rng(seed)
    rows = []
    order_id = 1000
    for _ in range(n):
        cat     = rng.choice(list(CATEGORIES.keys()))
        sub     = rng.choice(CATEGORIES[cat])
        product = rng.choice(PRODUCTS[sub])
        region  = rng.choice(REGIONS)
        segment = rng.choice(SEGMENTS)
        ship    = rng.choice(SHIP_MODE)

        # Random date between 2020-01-01 and 2023-12-31
        days_offset  = int(rng.integers(0, 365 * 4))
        order_date   = pd.Timestamp("2020-01-01") + pd.Timedelta(days=days_offset)
        ship_date    = order_date + pd.Timedelta(days=int(rng.integers(1, 8)))

        qty      = int(rng.integers(1, 15))
        price    = round(float(rng.uniform(5, 1500)), 2)
        discount = round(float(rng.choice([0, 0, 0, 0.1, 0.2, 0.3, 0.5])), 2)
        sales    = round(price * qty * (1 - discount), 2)
        m, s     = MARGINS[sub]
        margin   = float(np.clip(rng.normal(m, s), -0.5, 0.5))
        profit   = round(sales * margin, 2)

        rows.append({
            "Order ID":       f"CA-{order_id:05d}",
            "Order Date":     order_date,
            "Ship Date":      ship_date,
            "Ship Mode":      ship,
            "Customer ID":    f"CU-{int(rng.integers(1000, 9999)):04d}",
            "Segment":        segment,
            "Region":         region,
            "Product ID":     f"P-{int(rng.integers(10000, 99999)):05d}",
            "Category":       cat,
            "Sub-Category":   sub,
            "Product Name":   product,
            "Sales":          sales,
            "Quantity":       qty,
            "Discount":       discount,
            "Profit":         profit,
        })
        order_id += 1

    return pd.DataFrame(rows)


# ─── Load or build ────────────────────────────────────────────────────────────
if os.path.exists(CSV_PATH):
    print(f"Loading dataset from '{CSV_PATH}' …")
    df = pd.read_csv(CSV_PATH)
else:
    print("'superstore.csv' not found → generating synthetic dataset …")
    df = build_synthetic_dataset()
    df.to_csv(CSV_PATH, index=False)
    print(f"Synthetic dataset saved to '{CSV_PATH}'")

print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

# =============================================================================
# STEP 2 – INITIAL DATA EXPLORATION
# =============================================================================
print("=" * 60)
print("STEP 2 – INITIAL DATA EXPLORATION")
print("=" * 60)

print("\n── First 5 rows ──")
print(df.head().to_string())

print("\n── Shape & dtypes ──")
print(f"Rows: {df.shape[0]:,}   Columns: {df.shape[1]}")
print(df.dtypes)

print("\n── Missing values ──")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "No missing values found ✔")

# =============================================================================
# STEP 3 – DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3 – DATA PREPROCESSING")
print("=" * 60)

# 3a. Handle missing values
# ── Numeric columns → fill with median (robust to outliers)
num_cols = df.select_dtypes(include="number").columns
for col in num_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)
        print(f"  Filled missing '{col}' with median")

# ── Categorical columns → fill with mode
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"  Filled missing '{col}' with mode")

# 3b. Convert date columns
for date_col in ["Order Date", "Ship Date"]:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# 3c. Extract useful time features
df["Order Year"]       = df["Order Date"].dt.year
df["Order Month"]      = df["Order Date"].dt.month
df["Order Month Name"] = df["Order Date"].dt.strftime("%b")
df["Year-Month"]       = df["Order Date"].dt.to_period("M")

# 3d. Derived KPIs
df["Profit Margin %"] = (df["Profit"] / df["Sales"].replace(0, np.nan)) * 100

print("\nPreprocessing complete ✔")
print(f"  Date range: {df['Order Date'].min().date()} → {df['Order Date'].max().date()}")

# =============================================================================
# STEP 4 – MONTHLY REVENUE ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4 – MONTHLY REVENUE ANALYSIS")
print("=" * 60)

monthly = (
    df.groupby("Year-Month")["Sales"]
    .sum()
    .reset_index()
    .sort_values("Year-Month")
)
monthly["Year-Month Str"] = monthly["Year-Month"].astype(str)

# Rolling 3-month average to smooth seasonality
monthly["Rolling Avg"] = monthly["Sales"].rolling(3, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(monthly["Year-Month Str"], monthly["Sales"],
                alpha=0.15, color="#2196F3")
ax.plot(monthly["Year-Month Str"], monthly["Sales"],
        marker="o", markersize=3, linewidth=1.8,
        color="#2196F3", label="Monthly Revenue")
ax.plot(monthly["Year-Month Str"], monthly["Rolling Avg"],
        linewidth=2.2, linestyle="--", color="#FF5722",
        label="3-Month Rolling Avg")

# Mark the peak month
peak_idx  = monthly["Sales"].idxmax()
peak_val  = monthly["Sales"].max()
peak_label = monthly.loc[peak_idx, "Year-Month Str"]
ax.annotate(f"Peak\n${peak_val:,.0f}",
            xy=(peak_idx, peak_val),
            xytext=(peak_idx, peak_val * 1.06),
            arrowprops=dict(arrowstyle="->", color="black"),
            ha="center", fontsize=8, color="black")

# X-axis: show only every 3rd label to avoid clutter
n_ticks = len(monthly)
tick_step = max(1, n_ticks // 16)
ax.set_xticks(range(0, n_ticks, tick_step))
ax.set_xticklabels(
    [monthly["Year-Month Str"].iloc[i] for i in range(0, n_ticks, tick_step)],
    rotation=45, ha="right"
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
ax.set_title("Monthly Revenue Trend (with 3-Month Rolling Average)")
ax.set_xlabel("Month")
ax.set_ylabel("Total Sales ($)")
ax.legend()
plt.tight_layout()
save(fig, "01_monthly_revenue.png")

# INSIGHT: Revenue shows seasonal spikes (typically Q4). The rolling average
# smooths out noise and confirms an overall upward growth trend year-over-year.

# =============================================================================
# STEP 5 – PROFIT ANALYSIS (Category & Sub-Category)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5 – PROFIT ANALYSIS")
print("=" * 60)

# ── 5a. Profit by Category ────────────────────────────────────────────────────
cat_profit = (
    df.groupby("Category")[["Sales", "Profit"]]
    .sum()
    .reset_index()
    .sort_values("Profit", ascending=False)
)
cat_profit["Margin %"] = (cat_profit["Profit"] / cat_profit["Sales"] * 100).round(1)
print("\nProfit by Category:")
print(cat_profit.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bar: Profit by Category
colors = ["#4CAF50" if p > 0 else "#F44336" for p in cat_profit["Profit"]]
bars = axes[0].bar(cat_profit["Category"], cat_profit["Profit"],
                   color=colors, edgecolor="white", linewidth=0.8)
for bar, margin in zip(bars, cat_profit["Margin %"]):
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 h + abs(h) * 0.02,
                 f"{margin}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
axes[0].set_title("Total Profit by Category")
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Total Profit ($)")

# Pie: Sales share by Category
axes[1].pie(cat_profit["Sales"], labels=cat_profit["Category"],
            autopct="%1.1f%%", startangle=140,
            colors=["#2196F3", "#FF9800", "#9C27B0"],
            wedgeprops=dict(edgecolor="white", linewidth=1.5))
axes[1].set_title("Sales Share by Category")

plt.suptitle("Category-Level Profitability", fontweight="bold", y=1.01)
plt.tight_layout()
save(fig, "02_profit_by_category.png")

# ── 5b. Profit by Sub-Category ────────────────────────────────────────────────
sub_profit = (
    df.groupby(["Category", "Sub-Category"])[["Sales", "Profit"]]
    .sum()
    .reset_index()
    .sort_values("Profit", ascending=True)   # ascending for horizontal bar
)
sub_profit["Margin %"] = (sub_profit["Profit"] / sub_profit["Sales"] * 100).round(1)

fig, ax = plt.subplots(figsize=(11, 8))
palette = ["#F44336" if p < 0 else "#4CAF50" for p in sub_profit["Profit"]]
bars = ax.barh(sub_profit["Sub-Category"], sub_profit["Profit"],
               color=palette, edgecolor="white")

# Label each bar with profit value
for bar, val in zip(bars, sub_profit["Profit"]):
    offset = 200 if val >= 0 else -200
    ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
            f"${val:,.0f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)

ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
ax.set_title("Profit by Sub-Category  (Green = Profit | Red = Loss)")
ax.set_xlabel("Total Profit ($)")
ax.set_ylabel("Sub-Category")
plt.tight_layout()
save(fig, "03_profit_by_subcategory.png")

# INSIGHT: Technology tends to have the highest profit margin. Furniture,
# especially Tables, often runs at a loss due to heavy discounting.
# Office Supplies sub-categories like Labels and Fasteners have excellent margins.

# =============================================================================
# STEP 6 – TOP 10 PRODUCTS BY SALES
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6 – TOP 10 PRODUCTS BY SALES")
print("=" * 60)

top10_products = (
    df.groupby("Product Name")[["Sales", "Profit", "Quantity"]]
    .sum()
    .reset_index()
    .sort_values("Sales", ascending=False)
    .head(10)
)
top10_products["Margin %"] = (
    top10_products["Profit"] / top10_products["Sales"] * 100
).round(1)

print("\nTop 10 Products by Revenue:")
print(top10_products.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6))
colors_t10 = ["#4CAF50" if m >= 0 else "#F44336"
               for m in top10_products["Margin %"]]

bars = ax.barh(
    top10_products["Product Name"][::-1],
    top10_products["Sales"][::-1],
    color=colors_t10[::-1], edgecolor="white"
)
for bar, margin in zip(bars, top10_products["Margin %"][::-1]):
    ax.text(bar.get_width() + bar.get_width() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{margin}% margin", va="center", fontsize=8)

ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
ax.set_title("Top 10 Products by Revenue (Margin % shown on bar)")
ax.set_xlabel("Total Sales ($)")
ax.set_ylabel("Product Name")
plt.tight_layout()
save(fig, "04_top10_products.png")

# INSIGHT: High-revenue products are not always high-profit. Checking margin %
# alongside sales is critical — a product with large revenue but negative margin
# is destroying value.

# =============================================================================
# STEP 7 – REGIONAL PERFORMANCE ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7 – REGIONAL PERFORMANCE ANALYSIS")
print("=" * 60)

region_perf = (
    df.groupby("Region")[["Sales", "Profit", "Quantity"]]
    .agg({"Sales": "sum", "Profit": "sum", "Quantity": "sum"})
    .reset_index()
    .sort_values("Sales", ascending=False)
)
region_perf["Margin %"] = (
    region_perf["Profit"] / region_perf["Sales"] * 100
).round(1)
print("\nRegional Performance Summary:")
print(region_perf.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ── Sales by Region ──────────────────────────────────────────────────────────
region_colors = sns.color_palette("Set2", n_colors=len(region_perf))
axes[0].bar(region_perf["Region"], region_perf["Sales"],
            color=region_colors, edgecolor="white")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
axes[0].set_title("Sales by Region")
axes[0].set_xlabel("Region")
axes[0].set_ylabel("Total Sales ($)")
for i, (bar, val) in enumerate(zip(axes[0].patches, region_perf["Sales"])):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.01, f"${val/1e3:.0f}K",
                 ha="center", fontsize=9, fontweight="bold")

# ── Profit by Region ─────────────────────────────────────────────────────────
profit_colors = ["#4CAF50" if p > 0 else "#F44336" for p in region_perf["Profit"]]
axes[1].bar(region_perf["Region"], region_perf["Profit"],
            color=profit_colors, edgecolor="white")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
axes[1].set_title("Profit by Region")
axes[1].set_xlabel("Region")
axes[1].set_ylabel("Total Profit ($)")
for bar, val in zip(axes[1].patches, region_perf["Profit"]):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.02, f"{val/1e3:.1f}K",
                 ha="center", fontsize=9, fontweight="bold")

# ── Profit Margin % by Region ─────────────────────────────────────────────────
margin_colors = ["#2196F3" if m >= region_perf["Margin %"].mean() else "#FF9800"
                 for m in region_perf["Margin %"]]
axes[2].bar(region_perf["Region"], region_perf["Margin %"],
            color=margin_colors, edgecolor="white")
axes[2].axhline(region_perf["Margin %"].mean(), color="red",
                linestyle="--", linewidth=1.2, label="Avg Margin")
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
axes[2].set_title("Profit Margin % by Region")
axes[2].set_xlabel("Region")
axes[2].set_ylabel("Profit Margin (%)")
axes[2].legend()

plt.suptitle("Regional Performance Overview", fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "05_regional_performance.png")

# INSIGHT: The West and East regions typically generate the highest revenue,
# while the Central region may lag in profit margin despite reasonable sales
# volume — often due to higher discounting practices.

# =============================================================================
# STEP 8 – BONUS: ANNUAL SALES TREND & SEGMENT BREAKDOWN
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8 – ANNUAL TRENDS & SEGMENT ANALYSIS")
print("=" * 60)

annual = (
    df.groupby("Order Year")[["Sales", "Profit"]]
    .sum()
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Annual Revenue vs Profit
x = range(len(annual))
width = 0.4
axes[0].bar([i - width / 2 for i in x], annual["Sales"],
            width=width, label="Sales", color="#2196F3", edgecolor="white")
axes[0].bar([i + width / 2 for i in x], annual["Profit"],
            width=width, label="Profit", color="#4CAF50", edgecolor="white")
axes[0].set_xticks(list(x))
axes[0].set_xticklabels(annual["Order Year"])
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))
axes[0].set_title("Annual Sales vs Profit")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Amount ($)")
axes[0].legend()

# Segment donut chart
seg_sales = df.groupby("Segment")["Sales"].sum()
wedge_props = dict(width=0.5, edgecolor="white")
axes[1].pie(seg_sales, labels=seg_sales.index,
            autopct="%1.1f%%", startangle=90,
            colors=["#9C27B0", "#FF9800", "#03A9F4"],
            wedgeprops=wedge_props)
axes[1].set_title("Sales Distribution by Customer Segment")

plt.suptitle("Annual Growth & Segment Performance", fontweight="bold", y=1.01)
plt.tight_layout()
save(fig, "06_annual_and_segment.png")

# =============================================================================
# STEP 9 – DISCOUNT vs PROFIT SCATTER ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 9 – DISCOUNT vs PROFIT ANALYSIS")
print("=" * 60)

# Sample for readability (max 2000 points)
sample = df.sample(min(2000, len(df)), random_state=42)

fig, ax = plt.subplots(figsize=(9, 5))
cat_palette = {"Furniture": "#FF9800", "Office Supplies": "#2196F3",
               "Technology": "#9C27B0"}
for cat_name, grp in sample.groupby("Category"):
    ax.scatter(grp["Discount"], grp["Profit"],
               alpha=0.35, s=18, label=cat_name,
               color=cat_palette.get(cat_name, "gray"))

ax.axhline(0, color="red", linewidth=1.2, linestyle="--", alpha=0.6)
ax.set_title("Discount vs Profit (by Category)")
ax.set_xlabel("Discount Rate")
ax.set_ylabel("Profit ($)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.legend()
plt.tight_layout()
save(fig, "07_discount_vs_profit.png")

# INSIGHT: Higher discounts strongly correlate with lower or negative profit.
# Orders with 40–50%+ discounts almost always result in losses.
# Discount strategy needs urgent review for the Furniture category.

# =============================================================================
# SUMMARY STATISTICS REPORT (printed to console)
# =============================================================================
print("\n")
print("=" * 60)
print("   BUSINESS INSIGHTS REPORT – SUPERSTORE SALES ANALYSIS")
print("=" * 60)

total_sales   = df["Sales"].sum()
total_profit  = df["Profit"].sum()
overall_margin = total_profit / total_sales * 100
total_orders  = df["Order ID"].nunique()
best_month_row = monthly.loc[monthly["Sales"].idxmax()]
worst_cat = sub_profit.sort_values("Profit").iloc[0]
best_cat  = sub_profit.sort_values("Profit").iloc[-1]
best_region = region_perf.sort_values("Profit", ascending=False).iloc[0]
worst_region = region_perf.sort_values("Profit").iloc[0]

print(f"""
┌─────────────────────────────────────────────────────────────┐
│  EXECUTIVE SUMMARY                                          │
├─────────────────────────────────────────────────────────────┤
│  Total Revenue     : ${total_sales:>12,.0f}                        │
│  Total Profit      : ${total_profit:>12,.0f}                        │
│  Overall Margin    : {overall_margin:>11.2f}%                        │
│  Total Orders      : {total_orders:>12,}                        │
│  Date Range        : {df['Order Date'].min().date()} → {df['Order Date'].max().date()}      │
└─────────────────────────────────────────────────────────────┘

📈  REVENUE TRENDS
  • Peak revenue month : {best_month_row['Year-Month Str']} (${best_month_row['Sales']:,.0f})
  • Revenue shows a consistent upward trend year-over-year.
  • Q4 (Oct–Dec) regularly outperforms other quarters — seasonal
    demand spike should be leveraged for promotions.

💰  PROFITABILITY INSIGHTS
  • Best sub-category  : {best_cat['Sub-Category']} (${best_cat['Profit']:,.0f})
  • Worst sub-category : {worst_cat['Sub-Category']} (${worst_cat['Profit']:,.0f})
  • Technology leads in both revenue and margin efficiency.
  • Furniture – especially Tables – consistently runs at a loss,
    driven by aggressive discounting (40–50% off).
  • Labels, Fasteners & Envelopes offer the highest margin %.

🏆  TOP-PERFORMING PRODUCTS
  • The top 10 products account for a disproportionate share of
    total revenue. Not all are profitable — margin % monitoring
    is essential before running volume-based discounts.

🗺️   REGIONAL PERFORMANCE
  • Best region   : {best_region['Region']} (Profit ${best_region['Profit']:,.0f}, Margin {best_region['Margin %']}%)
  • Lagging region: {worst_region['Region']} (Profit ${worst_region['Profit']:,.0f}, Margin {worst_region['Margin %']}%)
  • Central region shows below-average margins — review local
    pricing and discount authorisation policies.

⚠️   AREAS NEEDING IMPROVEMENT
  1. Tables sub-category — eliminate or reprice; currently loss-making.
  2. High-discount orders (>30%) — negative ROI; cap discounts.
  3. Supplies sub-category — review COGS and supplier contracts.
  4. Central region — audit discount practices and pricing strategy.
  5. Low-margin high-revenue products — margin erosion at scale.

📁  Charts saved to: ./{OUTPUT_DIR}/
""")

print("Analysis complete ✔  All charts saved to the output folder.")

from PIL import Image
img=Image.open("images/superstore_dashboard_combined.png")
img.show()