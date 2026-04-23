"""
╔══════════════════════════════════════════════════════════════╗
║           INTERACTIVE DATA DASHBOARD - dashboard_app.py      ║
║   Upload any CSV and explore your data with dynamic charts   ║
╚══════════════════════════════════════════════════════════════╝

REQUIREMENTS:
    pip install streamlit pandas plotly

RUN:
    streamlit run dashboard_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the very first st call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DataLens Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME DEFINITIONS
# ─────────────────────────────────────────────
THEMES = {
    "Dark": {
        "bg": "#0f1117",
        "card": "#1c1f2e",
        "border": "#2d3149",
        "accent": "#6C63FF",
        "accent2": "#FF6584",
        "text": "#e8eaf0",
        "subtext": "#9094a8",
        "plotly_template": "plotly_dark",
        "font": "'IBM Plex Mono', monospace",
    },
    "Light": {
        "bg": "#f4f5fb",
        "card": "#ffffff",
        "border": "#dde0f0",
        "accent": "#4f46e5",
        "accent2": "#e11d48",
        "text": "#1e1f2e",
        "subtext": "#6b7280",
        "plotly_template": "plotly_white",
        "font": "'IBM Plex Mono', monospace",
    },
}

# ─────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

# ─────────────────────────────────────────────
# HELPER – inject CSS variables based on theme
# ─────────────────────────────────────────────
def inject_css(t: dict):
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@300;400;500;700&display=swap');

        /* ── Root & body ── */
        :root {{
            --bg:      {t['bg']};
            --card:    {t['card']};
            --border:  {t['border']};
            --accent:  {t['accent']};
            --accent2: {t['accent2']};
            --text:    {t['text']};
            --subtext: {t['subtext']};
        }}
        html, body, [class*="css"] {{
            background-color: var(--bg) !important;
            color: var(--text) !important;
            font-family: 'Space Grotesk', sans-serif !important;
        }}

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {{
            background: var(--card) !important;
            border-right: 1px solid var(--border);
        }}
        section[data-testid="stSidebar"] * {{
            color: var(--text) !important;
        }}
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stMultiSelect label,
        section[data-testid="stSidebar"] .stSlider label {{
            color: var(--subtext) !important;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        /* ── Metric cards ── */
        div[data-testid="metric-container"] {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            transition: border-color .2s;
        }}
        div[data-testid="metric-container"]:hover {{
            border-color: var(--accent);
        }}
        div[data-testid="metric-container"] label {{
            color: var(--subtext) !important;
            font-size: 0.7rem !important;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
            color: var(--accent) !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 1.6rem !important;
        }}

        /* ── Tabs ── */
        button[data-baseweb="tab"] {{
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.78rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: var(--accent) !important;
            border-bottom-color: var(--accent) !important;
        }}

        /* ── Dataframe ── */
        .stDataFrame {{ border-radius: 10px; overflow: hidden; }}
        .stDataFrame thead th {{
            background: var(--card) !important;
            color: var(--subtext) !important;
            font-size: 0.72rem;
            text-transform: uppercase;
        }}

        /* ── Headings ── */
        h1 {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: var(--text) !important; }}
        h2, h3 {{ font-family: 'IBM Plex Mono', monospace; color: var(--text) !important; }}

        /* ── Divider ── */
        hr {{ border-color: var(--border) !important; }}

        /* ── Upload widget ── */
        [data-testid="stFileUploadDropzone"] {{
            background: var(--card) !important;
            border: 2px dashed var(--border) !important;
            border-radius: 12px;
        }}
        [data-testid="stFileUploadDropzone"]:hover {{
            border-color: var(--accent) !important;
        }}

        /* ── Accent pill ── */
        .pill {{
            display: inline-block;
            background: var(--accent);
            color: #fff;
            font-size: 0.68rem;
            font-family: 'IBM Plex Mono', monospace;
            padding: 2px 10px;
            border-radius: 999px;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}

        /* ── Section header ── */
        .section-label {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--subtext);
            margin-bottom: 0.3rem;
        }}

        /* ── Insight card ── */
        .insight-card {{
            background: var(--card);
            border-left: 3px solid var(--accent);
            border-radius: 0 10px 10px 0;
            padding: 0.7rem 1rem;
            margin-bottom: 0.6rem;
            font-size: 0.85rem;
            color: var(--text);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# HELPER – detect column types
# ─────────────────────────────────────────────
def detect_columns(df: pd.DataFrame):
    """Return lists of numerical and categorical column names."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    return num_cols, cat_cols


# ─────────────────────────────────────────────
# HELPER – handle missing values
# ─────────────────────────────────────────────
def handle_missing(df: pd.DataFrame, strategy: str = "Drop rows") -> pd.DataFrame:
    """
    Fill or drop missing values based on user choice.
    - "Drop rows"   : remove any row with at least one NaN
    - "Fill with 0" : replace NaN with 0 in numeric columns
    - "Fill median" : replace NaN with column median (numeric)
    - "Keep as-is"  : do nothing
    """
    df = df.copy()
    if strategy == "Drop rows":
        df = df.dropna()
    elif strategy == "Fill with 0":
        num_cols, _ = detect_columns(df)
        df[num_cols] = df[num_cols].fillna(0)
    elif strategy == "Fill median":
        num_cols, _ = detect_columns(df)
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())
    # "Keep as-is" → no change
    return df


# ─────────────────────────────────────────────
# HELPER – auto insights
# ─────────────────────────────────────────────
def generate_insights(df: pd.DataFrame, num_cols: list, cat_cols: list) -> list[str]:
    insights = []
    insights.append(f"Dataset has **{df.shape[0]:,} rows** and **{df.shape[1]} columns**.")
    missing = df.isnull().sum().sum()
    if missing:
        insights.append(f"**{missing:,}** missing value(s) detected across the dataset.")
    else:
        insights.append("No missing values — your data is clean! ✓")
    for col in num_cols[:3]:  # top 3 numeric
        skew = df[col].skew()
        direction = "right-skewed (high outliers likely)" if skew > 0.5 else "left-skewed (low outliers likely)" if skew < -0.5 else "roughly symmetric"
        insights.append(f"**{col}** is {direction} (skew ≈ {skew:.2f}).")
    for col in cat_cols[:2]:  # top 2 categorical
        top_val = df[col].value_counts().idxmax()
        pct = df[col].value_counts(normalize=True).max() * 100
        insights.append(f"Most common **{col}**: `{top_val}` ({pct:.1f}% of rows).")
    return insights


# ─────────────────────────────────────────────
# HELPER – build Plotly chart
# ─────────────────────────────────────────────
def build_chart(chart_type: str, df: pd.DataFrame, x_col: str, y_col: str,
                color_col: str | None, template: str) -> go.Figure:
    """Return a Plotly figure based on chart_type and column selections."""
    kw = dict(template=template, height=420)

    if chart_type == "Bar Chart":
        if y_col:
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, barmode="group",
                         title=f"{y_col} by {x_col}", **kw)
        else:
            counts = df[x_col].value_counts().reset_index()
            counts.columns = [x_col, "count"]
            fig = px.bar(counts, x=x_col, y="count", title=f"Count of {x_col}", **kw)

    elif chart_type == "Line Chart":
        if y_col:
            fig = px.line(df, x=x_col, y=y_col, color=color_col,
                          title=f"{y_col} over {x_col}", markers=True, **kw)
        else:
            st.warning("Line chart needs both X and Y columns.")
            return go.Figure()

    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_col, color=color_col, nbins=40,
                           title=f"Distribution of {x_col}", **kw)

    elif chart_type == "Pie Chart":
        counts = df[x_col].value_counts().reset_index()
        counts.columns = [x_col, "count"]
        fig = px.pie(counts, names=x_col, values="count",
                     title=f"Proportions of {x_col}", **kw)

    elif chart_type == "Scatter Plot":
        if y_col:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                             title=f"{y_col} vs {x_col}", opacity=0.7, **kw)
        else:
            st.warning("Scatter plot needs both X and Y columns.")
            return go.Figure()

    elif chart_type == "Box Plot":
        if y_col:
            fig = px.box(df, x=x_col, y=y_col, color=color_col,
                         title=f"{y_col} distribution by {x_col}", **kw)
        else:
            fig = px.box(df, y=x_col, title=f"Box plot of {x_col}", **kw)

    else:
        return go.Figure()

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="Space Grotesk, sans-serif",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # ── Active theme ──
    theme = THEMES[st.session_state.theme]
    inject_css(theme)

    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        # Logo / title
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1.2rem;">
                <span style="font-family:'IBM Plex Mono',monospace;font-size:1.35rem;font-weight:600;">
                    📊 DataLens
                </span><br>
                <span style="font-size:0.72rem;opacity:0.5;letter-spacing:0.08em;">
                    INTERACTIVE DASHBOARD
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Theme toggle ──
        new_theme = st.radio(
            "🎨 Theme",
            ["Dark", "Light"],
            horizontal=True,
            index=0 if st.session_state.theme == "Dark" else 1,
        )
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()

        st.divider()

        # ── File upload ──
        st.markdown('<p class="section-label">📁 Data Source</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        missing_strategy = st.selectbox(
            "Missing value strategy",
            ["Drop rows", "Fill median", "Fill with 0", "Keep as-is"],
        )

        if uploaded:
            raw_df = pd.read_csv(uploaded)
            st.session_state.df_raw = raw_df

        st.divider()

        # ── Controls (only shown when data is loaded) ──
        if st.session_state.df_raw is not None:
            df_raw = st.session_state.df_raw
            df = handle_missing(df_raw, missing_strategy)
            num_cols, cat_cols = detect_columns(df)
            all_cols = df.columns.tolist()

            # ── Filters ──
            st.markdown('<p class="section-label">🔧 Filters</p>', unsafe_allow_html=True)

            # Categorical filters (multi-select)
            cat_filters = {}
            for col in cat_cols[:4]:  # limit to 4 categorical filters
                unique_vals = df[col].dropna().unique().tolist()
                if len(unique_vals) <= 30:  # only show if cardinality is reasonable
                    selected = st.multiselect(f"{col}", unique_vals, default=unique_vals)
                    cat_filters[col] = selected

            # Numeric range filters
            num_filters = {}
            for col in num_cols[:3]:  # limit to 3 numeric filters
                col_min = float(df[col].min())
                col_max = float(df[col].max())
                if col_min < col_max:
                    selected_range = st.slider(
                        f"{col} range",
                        min_value=col_min,
                        max_value=col_max,
                        value=(col_min, col_max),
                    )
                    num_filters[col] = selected_range

            # Apply filters
            df_filtered = df.copy()
            for col, vals in cat_filters.items():
                if vals:
                    df_filtered = df_filtered[df_filtered[col].isin(vals)]
            for col, (lo, hi) in num_filters.items():
                df_filtered = df_filtered[
                    (df_filtered[col] >= lo) & (df_filtered[col] <= hi)
                ]

            st.divider()

            # ── Chart config ──
            st.markdown('<p class="section-label">📈 Chart Config</p>', unsafe_allow_html=True)
            chart_type = st.selectbox(
                "Chart type",
                ["Bar Chart", "Line Chart", "Histogram", "Pie Chart", "Scatter Plot", "Box Plot"],
            )
            x_col = st.selectbox("X axis / primary column", all_cols)
            y_options = ["(none)"] + num_cols
            y_col_raw = st.selectbox("Y axis (numeric, optional)", y_options)
            y_col = None if y_col_raw == "(none)" else y_col_raw
            color_options = ["(none)"] + cat_cols
            color_raw = st.selectbox("Color / group by", color_options)
            color_col = None if color_raw == "(none)" else color_raw

            # ── Compare charts toggle ──
            st.divider()
            st.markdown('<p class="section-label">⚡ Bonus</p>', unsafe_allow_html=True)
            compare_mode = st.checkbox("Multi-chart comparison mode")

            if compare_mode:
                chart_type2 = st.selectbox(
                    "Chart 2 type",
                    ["Bar Chart", "Line Chart", "Histogram", "Pie Chart", "Scatter Plot", "Box Plot"],
                    key="ct2",
                )
                x_col2 = st.selectbox("Chart 2 X axis", all_cols, key="x2")
                y_col2_raw = st.selectbox("Chart 2 Y axis", y_options, key="y2")
                y_col2 = None if y_col2_raw == "(none)" else y_col2_raw
        else:
            df_filtered = None

    # ── Main area ─────────────────────────────
    if st.session_state.df_raw is None:
        # ── Landing / empty state ──
        st.markdown(
            """
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;height:60vh;text-align:center;gap:1rem;">
                <div style="font-size:4rem;">📊</div>
                <h1 style="font-size:2.2rem;margin:0;">DataLens Dashboard</h1>
                <p style="opacity:0.5;font-size:1rem;max-width:420px;">
                    Upload a CSV file from the sidebar to start exploring
                    your data through interactive charts and statistics.
                </p>
                <div style="display:flex;gap:0.6rem;flex-wrap:wrap;justify-content:center;margin-top:0.5rem;">
                    <span class="pill">Bar</span>
                    <span class="pill">Line</span>
                    <span class="pill">Histogram</span>
                    <span class="pill">Pie</span>
                    <span class="pill">Scatter</span>
                    <span class="pill">Box</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Header ──
    col_title, col_badge = st.columns([5, 1])
    with col_title:
        st.markdown(
            f"<h1 style='margin-bottom:0;'>Data Explorer</h1>"
            f"<p style='opacity:0.45;font-size:0.8rem;font-family:IBM Plex Mono,monospace;'>"
            f"Filtered: {len(df_filtered):,} / {len(df):,} rows  •  {df.shape[1]} columns</p>",
            unsafe_allow_html=True,
        )
    with col_badge:
        st.download_button(
            "⬇ Download CSV",
            data=df_filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()

    # ── KPI metrics ──
    kpi_cols = num_cols[:4] if num_cols else []
    if kpi_cols:
        cols = st.columns(len(kpi_cols))
        for i, col in enumerate(kpi_cols):
            with cols[i]:
                st.metric(
                    label=col,
                    value=f"{df_filtered[col].mean():,.2f}",
                    delta=f"median {df_filtered[col].median():,.2f}",
                )

    st.divider()

    # ── Tabs ──
    tab_chart, tab_stats, tab_data, tab_insights = st.tabs(
        ["📈 Charts", "📐 Statistics", "🗂 Dataset", "💡 Insights"]
    )

    # ── TAB 1 : CHARTS ──
    with tab_chart:
        if compare_mode:
            c1, c2 = st.columns(2)
            with c1:
                fig1 = build_chart(
                    chart_type, df_filtered, x_col, y_col, color_col,
                    theme["plotly_template"]
                )
                st.plotly_chart(fig1, use_container_width=True)
            with c2:
                fig2 = build_chart(
                    chart_type2, df_filtered, x_col2, y_col2, None,
                    theme["plotly_template"]
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            fig = build_chart(
                chart_type, df_filtered, x_col, y_col, color_col,
                theme["plotly_template"]
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Correlation heatmap (bonus for numeric datasets) ──
        if len(num_cols) >= 3:
            with st.expander("🔥 Correlation Heatmap"):
                corr = df_filtered[num_cols].corr()
                heat = px.imshow(
                    corr,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    template=theme["plotly_template"],
                    title="Pearson Correlation Matrix",
                    height=420,
                )
                heat.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(heat, use_container_width=True)

    # ── TAB 2 : STATISTICS ──
    with tab_stats:
        if num_cols:
            st.markdown("### Numerical Summary")
            desc = df_filtered[num_cols].describe().T
            desc["skew"] = df_filtered[num_cols].skew()
            desc["missing"] = df_filtered[num_cols].isnull().sum()
            st.dataframe(desc.style.format("{:.3f}"), use_container_width=True)
        if cat_cols:
            st.markdown("### Categorical Summary")
            cat_summary = pd.DataFrame({
                col: [
                    df_filtered[col].nunique(),
                    df_filtered[col].value_counts().idxmax(),
                    f"{df_filtered[col].value_counts(normalize=True).max()*100:.1f}%",
                    df_filtered[col].isnull().sum(),
                ]
                for col in cat_cols
            }, index=["Unique values", "Top value", "Top value %", "Missing"]).T
            st.dataframe(cat_summary, use_container_width=True)

    # ── TAB 3 : DATASET ──
    with tab_data:
        rows_to_show = st.slider("Rows to preview", 5, min(200, len(df_filtered)), 20)
        st.dataframe(df_filtered.head(rows_to_show), use_container_width=True, height=420)

    # ── TAB 4 : INSIGHTS ──
    with tab_insights:
        insights = generate_insights(df_filtered, num_cols, cat_cols)
        for insight in insights:
            st.markdown(
                f'<div class="insight-card">• {insight}</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()