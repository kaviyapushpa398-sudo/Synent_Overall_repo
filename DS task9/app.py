"""
=============================================================
  🏠 California House Price Predictor
  End-to-End Data Science Project with Streamlit UI
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
import os
import warnings
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background: #0f1117; }

    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        font-weight: 400;
        color: #f5f0e8;
        line-height: 1.15;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #8b8fa8;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 0.02em;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #16192a 100%);
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 600;
        color: #e2c97e;
        font-family: 'DM Serif Display', serif;
    }
    .metric-unit {
        font-size: 0.85rem;
        color: #8b8fa8;
    }

    .prediction-box {
        background: linear-gradient(135deg, #1c2a1c 0%, #162016 100%);
        border: 1.5px solid #3a6b3a;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .prediction-label {
        font-size: 0.9rem;
        color: #6b9b6b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    .prediction-value {
        font-family: 'DM Serif Display', serif;
        font-size: 3.2rem;
        color: #7eca7e;
        line-height: 1.1;
    }
    .prediction-range {
        font-size: 0.9rem;
        color: #5a8a5a;
        margin-top: 0.4rem;
    }

    .insight-card {
        background: #1a1d2e;
        border-left: 3px solid #e2c97e;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
        color: #c8ccd8;
    }

    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.5rem;
        color: #f5f0e8;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #2a2d3e;
    }

    .stSelectbox > div > div { background: #1a1d2e; }
    .stSlider { padding: 0.5rem 0; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1a1d2e;
        border-radius: 8px;
        color: #8b8fa8;
        border: 1px solid #2a2d3e;
        padding: 0.4rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background: #252840 !important;
        color: #e2c97e !important;
        border-color: #e2c97e !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d0f1a;
        border-right: 1px solid #1f2133;
    }

    .badge {
        display: inline-block;
        background: #252840;
        border: 1px solid #3a3e5c;
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.75rem;
        color: #a0a4c0;
        margin: 0.15rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  DATA FUNCTIONS
# ═══════════════════════════════════════════════

@st.cache_data
def load_and_clean_data():
    """Load California Housing dataset and perform cleaning steps."""
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()

    # Rename columns for readability
    df.columns = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal"
    ]

    # ── Cleaning Step 1: Check & report missing values
    missing_before = df.isnull().sum().sum()

    # ── Cleaning Step 2: Remove exact duplicates
    dupes_before = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)

    # ── Cleaning Step 3: Cap outliers at 99.5th percentile
    cap_cols = ["AveRooms", "AveBedrms", "Population", "AveOccup"]
    for col in cap_cols:
        upper = df[col].quantile(0.995)
        df[col] = df[col].clip(upper=upper)

    # ── Cleaning Step 4: Target in $1000s → dollars
    df["MedHouseVal"] = df["MedHouseVal"] * 100_000

    cleaning_report = {
        "rows_loaded": len(df) + dupes_before,
        "missing_values": int(missing_before),
        "duplicates_removed": int(dupes_before),
        "outliers_capped": cap_cols,
        "rows_clean": len(df),
    }

    return df, cleaning_report


@st.cache_resource
def train_and_save_model(df):
    """Train RandomForest model, evaluate, and cache."""
    FEATURE_COLS = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                    "Population", "AveOccup", "Latitude", "Longitude"]
    TARGET = "MedHouseVal"

    X = df[FEATURE_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=150, max_depth=15,
                                            random_state=42, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(n_estimators=150, learning_rate=0.1,
                                                max_depth=5, random_state=42))
        ]),
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
    }

    results = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        results[name] = {
            "pipeline": pipeline,
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "mae":  mean_absolute_error(y_test, preds),
            "r2":   r2_score(y_test, preds),
            "preds": preds,
            "y_test": y_test,
        }

    # Save best model (Random Forest) to pickle
    best_pipeline = results["Random Forest"]["pipeline"]
    with open("best_model.pkl", "wb") as f:
        pickle.dump({"model": best_pipeline, "features": FEATURE_COLS}, f)

    return results, X_train, X_test, y_train, y_test, FEATURE_COLS


def load_saved_model():
    """Load pickled model if it exists."""
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            return pickle.load(f)
    return None


# ═══════════════════════════════════════════════
#  VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════

def plot_distribution(df):
    """Target variable distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#0f1117")

    for ax in axes:
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="#8b8fa8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3e")

    vals = df["MedHouseVal"] / 1000  # in $K

    axes[0].hist(vals, bins=50, color="#e2c97e", alpha=0.85, edgecolor="#0f1117", linewidth=0.4)
    axes[0].set_title("Distribution of House Prices", color="#f5f0e8", fontsize=12, pad=10)
    axes[0].set_xlabel("Price ($K)", color="#8b8fa8")
    axes[0].set_ylabel("Count", color="#8b8fa8")
    axes[0].axvline(vals.median(), color="#7eca7e", linewidth=1.5, linestyle="--",
                    label=f"Median: ${vals.median():.0f}K")
    axes[0].legend(fontsize=9, facecolor="#1a1d2e", labelcolor="#c8ccd8", edgecolor="#2a2d3e")

    axes[1].boxplot(vals, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#252840", color="#e2c97e"),
                    whiskerprops=dict(color="#8b8fa8"),
                    capprops=dict(color="#e2c97e"),
                    medianprops=dict(color="#7eca7e", linewidth=2),
                    flierprops=dict(marker="o", color="#e2c97e", alpha=0.3, markersize=3))
    axes[1].set_title("Box Plot — House Prices", color="#f5f0e8", fontsize=12, pad=10)
    axes[1].set_ylabel("Price ($K)", color="#8b8fa8")
    axes[1].set_xticklabels(["MedHouseVal"])

    plt.tight_layout(pad=1.5)
    return fig


def plot_correlation(df):
    """Correlation heatmap."""
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True,
                fmt=".2f", ax=ax, linewidths=0.5, linecolor="#0f1117",
                annot_kws={"size": 9, "color": "#f5f0e8"},
                cbar_kws={"shrink": 0.8})

    ax.set_title("Feature Correlation Matrix", color="#f5f0e8", fontsize=13, pad=14)
    ax.tick_params(colors="#a0a4c0", labelsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_feature_vs_price(df):
    """Top 4 feature scatter plots vs price."""
    top_features = ["MedInc", "HouseAge", "AveRooms", "AveOccup"]
    colors = ["#e2c97e", "#7eca7e", "#7eb8ec", "#ec7e9e"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor("#0f1117")
    axes = axes.flatten()

    sample = df.sample(3000, random_state=42)

    for i, (feat, col) in enumerate(zip(top_features, colors)):
        ax = axes[i]
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="#8b8fa8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3e")

        ax.scatter(sample[feat], sample["MedHouseVal"] / 1000,
                   alpha=0.25, s=10, color=col)
        ax.set_xlabel(feat, color="#8b8fa8", fontsize=10)
        ax.set_ylabel("Price ($K)", color="#8b8fa8", fontsize=10)
        ax.set_title(f"{feat} vs Price", color="#f5f0e8", fontsize=11, pad=8)

        z = np.polyfit(sample[feat], sample["MedHouseVal"] / 1000, 1)
        p = np.poly1d(z)
        x_line = np.linspace(sample[feat].min(), sample[feat].max(), 100)
        ax.plot(x_line, p(x_line), color=col, linewidth=1.8, alpha=0.8)

    plt.tight_layout(pad=2)
    return fig


def plot_geographic(df):
    """Geographic scatter of prices."""
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")

    scatter = ax.scatter(
        df["Longitude"], df["Latitude"],
        c=df["MedHouseVal"] / 1000,
        cmap="YlOrRd", alpha=0.4, s=3,
        vmin=0, vmax=500
    )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Price ($K)", color="#a0a4c0", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#a0a4c0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#a0a4c0")

    ax.set_title("California — House Prices by Location", color="#f5f0e8",
                 fontsize=12, pad=10)
    ax.set_xlabel("Longitude", color="#8b8fa8")
    ax.set_ylabel("Latitude", color="#8b8fa8")
    ax.tick_params(colors="#8b8fa8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3e")

    plt.tight_layout()
    return fig


def plot_model_comparison(results):
    """Compare all models side-by-side."""
    names = list(results.keys())
    rmses = [results[n]["rmse"] / 1000 for n in names]
    r2s   = [results[n]["r2"]   for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#0f1117")
    colors = ["#e2c97e", "#7eca7e", "#7eb8ec"]

    for ax in axes:
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="#8b8fa8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3e")

    bars1 = axes[0].bar(names, rmses, color=colors, width=0.5, edgecolor="#0f1117")
    axes[0].set_title("RMSE by Model ($K)", color="#f5f0e8", fontsize=12, pad=10)
    axes[0].set_ylabel("RMSE ($K)", color="#8b8fa8")
    for bar, val in zip(bars1, rmses):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1, f"${val:.1f}K",
                     ha="center", va="bottom", color="#f5f0e8", fontsize=9)

    bars2 = axes[1].bar(names, r2s, color=colors, width=0.5, edgecolor="#0f1117")
    axes[1].set_title("R² Score by Model", color="#f5f0e8", fontsize=12, pad=10)
    axes[1].set_ylabel("R²", color="#8b8fa8")
    axes[1].set_ylim(0, 1)
    for bar, val in zip(bars2, r2s):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01, f"{val:.3f}",
                     ha="center", va="bottom", color="#f5f0e8", fontsize=9)

    plt.tight_layout(pad=1.5)
    return fig


def plot_actual_vs_predicted(results, model_name="Random Forest"):
    """Actual vs Predicted scatter."""
    preds  = results[model_name]["preds"]  / 1000
    actual = results[model_name]["y_test"] / 1000

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")

    ax.scatter(actual, preds, alpha=0.25, s=8, color="#e2c97e")
    lims = [min(actual.min(), preds.min()), max(actual.max(), preds.max())]
    ax.plot(lims, lims, color="#7eca7e", linewidth=1.5, linestyle="--", label="Perfect Fit")
    ax.set_xlabel("Actual Price ($K)", color="#8b8fa8", fontsize=11)
    ax.set_ylabel("Predicted Price ($K)", color="#8b8fa8", fontsize=11)
    ax.set_title(f"{model_name} — Actual vs Predicted", color="#f5f0e8", fontsize=12, pad=10)
    ax.tick_params(colors="#8b8fa8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3e")
    ax.legend(facecolor="#1a1d2e", labelcolor="#c8ccd8", edgecolor="#2a2d3e")

    r2 = results[model_name]["r2"]
    ax.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax.transAxes,
            color="#e2c97e", fontsize=11)

    plt.tight_layout()
    return fig


def plot_feature_importance(results, feature_cols):
    """Feature importance from Random Forest."""
    rf_model = results["Random Forest"]["pipeline"].named_steps["model"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")

    colors_fi = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(feature_cols)))

    bars = ax.barh([feature_cols[i] for i in indices],
                   [importances[i] for i in indices],
                   color=colors_fi, edgecolor="#0f1117")

    ax.set_title("Feature Importance — Random Forest", color="#f5f0e8", fontsize=12, pad=10)
    ax.set_xlabel("Importance", color="#8b8fa8")
    ax.tick_params(colors="#a0a4c0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3e")

    for bar, val in zip(bars, [importances[i] for i in indices]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="#f5f0e8", fontsize=9)

    plt.tight_layout()
    return fig


def plot_residuals(results, model_name="Random Forest"):
    """Residual distribution."""
    preds  = results[model_name]["preds"]
    actual = np.array(results[model_name]["y_test"])
    residuals = (actual - preds) / 1000

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#0f1117")

    for ax in axes:
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="#8b8fa8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3e")

    axes[0].hist(residuals, bins=60, color="#7eb8ec", alpha=0.85, edgecolor="#0f1117", linewidth=0.3)
    axes[0].axvline(0, color="#e2c97e", linewidth=1.5, linestyle="--")
    axes[0].set_title("Residual Distribution", color="#f5f0e8", fontsize=12, pad=10)
    axes[0].set_xlabel("Residual ($K)", color="#8b8fa8")
    axes[0].set_ylabel("Count", color="#8b8fa8")

    preds_k = preds / 1000
    axes[1].scatter(preds_k, residuals, alpha=0.2, s=7, color="#7eb8ec")
    axes[1].axhline(0, color="#e2c97e", linewidth=1.5, linestyle="--")
    axes[1].set_title("Residuals vs Fitted", color="#f5f0e8", fontsize=12, pad=10)
    axes[1].set_xlabel("Fitted Value ($K)", color="#8b8fa8")
    axes[1].set_ylabel("Residual ($K)", color="#8b8fa8")

    plt.tight_layout(pad=1.5)
    return fig


# ═══════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════

def main():
    # ── SIDEBAR ──────────────────────────────
    with st.sidebar:
        st.markdown("### 🏠 House Price Predictor")
        st.markdown("---")
        st.markdown("""
        **Dataset:** California Housing  
        **Source:** Scikit-learn built-in  
        **Rows:** ~20,640  
        **Target:** Median House Value
        """)
        st.markdown("---")
        st.markdown("**Tech Stack**")
        for badge in ["Python 3.10+", "Scikit-learn", "Streamlit", "Pandas", "Seaborn", "Pickle"]:
            st.markdown(f'<span class="badge">{badge}</span>', unsafe_allow_html=True)
        st.markdown("---")
        st.caption("Built as an end-to-end Data Science project demo.")

    # ── HERO ─────────────────────────────────
    st.markdown('<p class="hero-title">California House Price Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">End-to-End Machine Learning · EDA · Random Forest · Interactive Predictions</p>',
                unsafe_allow_html=True)

    # ── LOAD DATA ────────────────────────────
    with st.spinner("Loading and cleaning dataset..."):
        df, cleaning_report = load_and_clean_data()

    with st.spinner("Training models (cached after first run)..."):
        results, X_train, X_test, y_train, y_test, feature_cols = train_and_save_model(df)

    saved = load_saved_model()

    # ── TOP KPI METRICS ──────────────────────
    rf = results["Random Forest"]
    col1, col2, col3, col4, col5 = st.columns(5)

    kpis = [
        ("Dataset Rows", f"{cleaning_report['rows_clean']:,}", "records"),
        ("Features Used", "8", "input vars"),
        ("Model R² Score", f"{rf['r2']:.4f}", "Random Forest"),
        ("RMSE", f"${rf['rmse']/1000:.1f}K", "avg error"),
        ("MAE", f"${rf['mae']/1000:.1f}K", "avg abs error"),
    ]
    for col, (label, value, unit) in zip([col1, col2, col3, col4, col5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📂 Data Overview",
        "📊 EDA & Insights",
        "🤖 Model Performance",
        "🔮 Make a Prediction",
        "🔧 Data Cleaning Log",
    ])

    # ═══════════════════════════════
    # TAB 1 — DATA OVERVIEW
    # ═══════════════════════════════
    with tab1:
        st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("**Sample Records (first 10 rows)**")
            display_df = df.head(10).copy()
            display_df["MedHouseVal"] = display_df["MedHouseVal"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(display_df, use_container_width=True)

        with c2:
            st.markdown("**Column Types**")
            dtypes_df = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.values.astype(str)
            })
            st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

        st.markdown('<p class="section-header">Summary Statistics</p>', unsafe_allow_html=True)
        stats = df.describe().T.round(3)
        st.dataframe(stats, use_container_width=True)

        st.markdown('<p class="section-header">Feature Descriptions</p>', unsafe_allow_html=True)
        descriptions = {
            "MedInc":     "Median income in the block group (in tens of thousands of USD)",
            "HouseAge":   "Median house age in the block group (years)",
            "AveRooms":   "Average number of rooms per household",
            "AveBedrms":  "Average number of bedrooms per household",
            "Population": "Block group population",
            "AveOccup":   "Average number of household members",
            "Latitude":   "Block group latitude coordinate",
            "Longitude":  "Block group longitude coordinate",
            "MedHouseVal":"Median house value (TARGET — in USD)",
        }
        desc_df = pd.DataFrame({"Feature": descriptions.keys(), "Description": descriptions.values()})
        st.dataframe(desc_df, use_container_width=True, hide_index=True)

    # ═══════════════════════════════
    # TAB 2 — EDA
    # ═══════════════════════════════
    with tab2:
        st.markdown('<p class="section-header">Exploratory Data Analysis</p>', unsafe_allow_html=True)

        # Insights
        median_price = df["MedHouseVal"].median()
        high_inc_corr = df["MedInc"].corr(df["MedHouseVal"])

        insights = [
            f"📍 Median house price is <b>${median_price:,.0f}</b> across California block groups.",
            f"💰 Median Income (MedInc) has the strongest correlation with price: <b>r = {high_inc_corr:.3f}</b>.",
            f"🗓️ House age has a weak negative correlation — newer isn't always more expensive.",
            f"🌊 Coastal areas (LA / Bay Area) clearly command price premiums visible on the geo map.",
            f"🏘️ Over-occupied blocks (high AveOccup) tend to have lower house values.",
        ]
        for ins in insights:
            st.markdown(f'<div class="insight-card">💡 {ins}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**Price Distribution**")
        st.pyplot(plot_distribution(df))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Correlation Heatmap**")
            st.pyplot(plot_correlation(df))
        with c2:
            st.markdown("**Geographic Price Map**")
            st.pyplot(plot_geographic(df))

        st.markdown("**Top Features vs Price**")
        st.pyplot(plot_feature_vs_price(df))

    # ═══════════════════════════════
    # TAB 3 — MODEL PERFORMANCE
    # ═══════════════════════════════
    with tab3:
        st.markdown('<p class="section-header">Model Comparison & Evaluation</p>', unsafe_allow_html=True)

        # Model metrics table
        metrics_data = []
        for name, res in results.items():
            metrics_data.append({
                "Model": name,
                "RMSE ($K)": f"${res['rmse']/1000:.2f}K",
                "MAE ($K)":  f"${res['mae']/1000:.2f}K",
                "R² Score":  f"{res['r2']:.4f}",
                "Accuracy*": f"{min(100, res['r2']*100):.1f}%",
            })
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.caption("*Approximate accuracy based on R² × 100. Best deployment model: **Random Forest**.")

        st.pyplot(plot_model_comparison(results))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Actual vs Predicted — Random Forest**")
            st.pyplot(plot_actual_vs_predicted(results, "Random Forest"))
        with c2:
            st.markdown("**Feature Importance**")
            st.pyplot(plot_feature_importance(results, feature_cols))

        st.markdown("**Residual Analysis — Random Forest**")
        st.pyplot(plot_residuals(results, "Random Forest"))

        if saved:
            st.success("✅ Model saved to `best_model.pkl` using pickle.")

    # ═══════════════════════════════
    # TAB 4 — PREDICTION
    # ═══════════════════════════════
    with tab4:
        st.markdown('<p class="section-header">🔮 Predict House Price</p>', unsafe_allow_html=True)
        st.markdown("Adjust the sliders below to match the property details, then click **Predict**.")

        c_left, c_right = st.columns([2, 1])

        with c_left:
            st.markdown("**Economic & Structural Features**")
            r1, r2 = st.columns(2)

            with r1:
                med_inc    = st.slider("Median Income (×$10K)", 0.5, 15.0, 4.5, 0.1,
                                       help="Median income in the block group")
                house_age  = st.slider("House Age (years)", 1, 52, 20,
                                       help="Median age of houses in the block")
                ave_rooms  = st.slider("Average Rooms", 1.0, 12.0, 5.2, 0.1,
                                       help="Avg rooms per household")
                ave_bedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.05, 0.05,
                                       help="Avg bedrooms per household")
            with r2:
                population = st.slider("Block Population", 50, 5000, 1200,
                                       help="Total population in block group")
                ave_occup  = st.slider("Average Occupancy", 1.0, 10.0, 2.8, 0.1,
                                       help="Avg number of people per household")

            st.markdown("**Location**")
            lc1, lc2 = st.columns(2)
            with lc1:
                latitude  = st.slider("Latitude", 32.5, 42.0, 37.5, 0.01,
                                      help="Northern California ≈ 38–42, Southern ≈ 32–36")
            with lc2:
                longitude = st.slider("Longitude", -124.5, -114.0, -122.0, 0.01,
                                      help="Coastal ≈ -122 to -120, Inland ≈ -117 to -114")

        with c_right:
            st.markdown("**Input Summary**")
            summary_df = pd.DataFrame({
                "Feature": feature_cols,
                "Value": [med_inc, house_age, ave_rooms, ave_bedrms,
                          population, ave_occup, latitude, longitude]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮  Predict House Price", use_container_width=False,
                                type="primary")

        if predict_btn:
            user_input = np.array([[med_inc, house_age, ave_rooms, ave_bedrms,
                                    population, ave_occup, latitude, longitude]])

            best_pipeline = results["Random Forest"]["pipeline"]
            prediction = best_pipeline.predict(user_input)[0]

            # Uncertainty range (±1 std from training residuals)
            train_preds   = best_pipeline.predict(X_train)
            train_residuals = y_train - train_preds
            sigma = train_residuals.std()
            lower = max(0, prediction - sigma)
            upper = prediction + sigma

            pred_k  = prediction / 1000
            lower_k = lower / 1000
            upper_k = upper / 1000

            col_pred, col_gauge = st.columns([1, 1])

            with col_pred:
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-label">Estimated House Value</div>
                    <div class="prediction-value">${pred_k:,.1f}K</div>
                    <div class="prediction-range">
                        Confidence Range: ${lower_k:,.1f}K – ${upper_k:,.1f}K
                    </div>
                </div>""", unsafe_allow_html=True)

            with col_gauge:
                # Mini gauge chart
                fig_g, ax_g = plt.subplots(figsize=(5, 3.5))
                fig_g.patch.set_facecolor("#0f1117")
                ax_g.set_facecolor("#0f1117")

                categories = ["< $150K", "$150K–300K", "$300K–450K", "> $450K"]
                cat_colors = ["#7eb8ec", "#7eca7e", "#e2c97e", "#ec7e9e"]
                cat_maxes  = [150, 300, 450, 600]

                bar_heights = [1] * 4
                bars = ax_g.bar(categories, bar_heights, color=cat_colors, edgecolor="#0f1117")

                # Highlight predicted category
                cat_idx = min(3, int(pred_k // 150))
                bars[cat_idx].set_edgecolor("#ffffff")
                bars[cat_idx].set_linewidth(2.5)
                ax_g.text(cat_idx, 1.05, "▲ Your\nPrediction",
                          ha="center", va="bottom", color="#f5f0e8", fontsize=8.5)

                ax_g.set_ylim(0, 1.4)
                ax_g.set_title("Price Category", color="#f5f0e8", fontsize=11, pad=8)
                ax_g.set_yticklabels([])
                ax_g.tick_params(colors="#8b8fa8", labelsize=8)
                for spine in ax_g.spines.values():
                    spine.set_edgecolor("#2a2d3e")
                plt.tight_layout()
                st.pyplot(fig_g)

            # Interpretation
            st.markdown("---")
            st.markdown("**🔍 Prediction Insights**")
            i1, i2, i3 = st.columns(3)
            with i1:
                avg_price = df["MedHouseVal"].mean()
                diff = ((prediction - avg_price) / avg_price) * 100
                st.info(f"{'Above' if diff > 0 else 'Below'} CA average by **{abs(diff):.1f}%**")
            with i2:
                pct_rank = (df["MedHouseVal"] < prediction).mean() * 100
                st.info(f"Higher than **{pct_rank:.0f}%** of California homes")
            with i3:
                income_factor = round(prediction / (med_inc * 10000), 1)
                st.info(f"Price-to-income ratio: **{income_factor}×**")

    # ═══════════════════════════════
    # TAB 5 — CLEANING LOG
    # ═══════════════════════════════
    with tab5:
        st.markdown('<p class="section-header">Data Cleaning Pipeline Log</p>', unsafe_allow_html=True)

        steps = [
            ("✅ Step 1: Load Dataset", f"Loaded **{cleaning_report['rows_loaded']:,}** rows from California Housing (sklearn)."),
            ("✅ Step 2: Missing Value Check", f"Found **{cleaning_report['missing_values']}** missing values — dataset is clean."),
            ("✅ Step 3: Duplicate Removal", f"Removed **{cleaning_report['duplicates_removed']}** duplicate rows."),
            ("✅ Step 4: Outlier Capping", f"Capped top 0.5% of: `{'`, `'.join(cleaning_report['outliers_capped'])}`"),
            ("✅ Step 5: Unit Conversion", "Converted target from `$100K units` → actual USD values (×100,000)."),
            ("✅ Step 6: Train/Test Split", "80% train / 20% test, `random_state=42` for reproducibility."),
            ("✅ Step 7: Feature Scaling", "StandardScaler applied inside sklearn Pipeline (no data leakage)."),
            ("✅ Step 8: Model Saved", "Best model (Random Forest) pickled to `best_model.pkl`."),
        ]

        for title, desc in steps:
            with st.expander(title, expanded=True):
                st.markdown(desc)

        st.markdown('<p class="section-header">Final Dataset Shape</p>', unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Rows", f"{cleaning_report['rows_clean']:,}")
        col_b.metric("Features", "8")
        col_c.metric("Target Variable", "MedHouseVal")


if __name__ == "__main__":
    main()