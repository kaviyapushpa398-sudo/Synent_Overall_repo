# =============================================================================
# STOCK PRICE TIME SERIES ANALYSIS
# Professional Data Science & Forecasting Script
# Tools: pandas, matplotlib, seaborn, statsmodels
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os

warnings.filterwarnings("ignore")

# ── Optional heavy imports (gracefully skipped if not installed) ──────────────
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("statsmodels not found — decomposition & ARIMA will be skipped.")
    print("Install with:  pip install statsmodels\n")

# =============================================================================
# GLOBAL STYLE
# =============================================================================
DARK_BG    = "#0A0E1A"
CARD_BG    = "#111827"
GRID_COL   = "#1F2937"
TEXT_COL   = "#E5E7EB"
UP_COLOR   = "#00D68F"   # green  – bullish
DOWN_COLOR = "#FF4757"   # red    – bearish
VOL_COLOR  = "#3B82F6"   # blue   – volume
GOLD       = "#F59E0B"
PURPLE     = "#8B5CF6"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "text.color":        TEXT_COL,
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.5,
    "axes.grid":         True,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "legend.framealpha": 0.2,
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
})

OUTPUT_DIR = "stock_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  ✔  Saved → {path}")
    plt.close(fig)

def section(title):
    print(f"\n{'='*62}\n  {title}\n{'='*62}")

# =============================================================================
# STEP 1 — GENERATE / LOAD DATASET
# Loads 'stock_prices.csv' if present; otherwise synthesises a realistic
# 5-year OHLCV dataset with trend + seasonality + volatility clusters.
# =============================================================================

CSV_PATH   = "stock_prices.csv"
TICKER     = "TECHX"

def build_synthetic(days=1260, seed=42):
    """
    Generates realistic daily OHLCV data using:
      - Long-term upward drift
      - Seasonal sine-wave component (annual cycle)
      - GARCH-like volatility clustering
      - Occasional shock events
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=days)  # business days only

    # Geometric Brownian Motion parameters
    mu       = 0.0004    # daily drift  (~10 % annual)
    sigma    = 0.018     # base daily volatility
    S0       = 150.0     # starting price

    prices   = [S0]
    vol      = sigma

    for i in range(1, days):
        # Volatility clustering (GARCH-lite)
        shock = abs(rng.normal(0, 1))
        vol   = 0.94 * vol + 0.06 * sigma * shock

        # Seasonal component: higher prices in Q4
        season = 0.0003 * np.sin(2 * np.pi * i / 252)

        # Rare shock events (earnings surprise / macro event)
        event  = rng.choice([0, 0, 0, 0, 0, 0, 0, 0,
                              rng.uniform(-0.08, 0.12)],
                            p=[0.9, 0.02, 0.02, 0.02, 0.01,
                               0.01, 0.01, 0.0025, 0.0075])

        ret    = mu + season + vol * rng.normal() + event
        prices.append(prices[-1] * np.exp(ret))

    prices = np.array(prices)

    # Build OHLCV from close prices
    daily_range = prices * rng.uniform(0.005, 0.025, size=days)
    open_  = prices * (1 + rng.uniform(-0.008, 0.008, size=days))
    high   = prices + daily_range
    low    = prices - daily_range
    volume = (rng.integers(500_000, 5_000_000, size=days)
              * (1 + 2 * (abs(prices - np.roll(prices, 1)) / prices)))

    df = pd.DataFrame({
        "Date":   dates,
        "Open":   open_.round(2),
        "High":   high.round(2),
        "Low":    low.round(2),
        "Close":  prices.round(2),
        "Volume": volume.astype(int),
    })
    return df

section("STEP 1 — LOAD DATASET")
if os.path.exists(CSV_PATH):
    print(f"Loading '{CSV_PATH}' …")
    raw = pd.read_csv(CSV_PATH)
else:
    print(f"'{CSV_PATH}' not found → generating synthetic {TICKER} dataset …")
    raw = build_synthetic()
    raw.to_csv(CSV_PATH, index=False)
    print(f"Saved synthetic data as '{CSV_PATH}'")

print(f"Loaded : {len(raw):,} rows × {raw.shape[1]} columns")
print(raw.head().to_string())

# =============================================================================
# STEP 2 — DATA PREPROCESSING
# =============================================================================
section("STEP 2 — DATA PREPROCESSING")

df = raw.copy()

# 2a. Convert date and set as index
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)

# 2b. Handle missing values
missing = df.isnull().sum()
print("\nMissing values before treatment:")
print(missing[missing > 0] if missing.any() else "  None found ✔")

# Forward-fill gaps (e.g., holidays), then backfill any leading NaN
df.ffill(inplace=True)
df.bfill(inplace=True)

# 2c. Derived columns
df["Daily Return %"]    = df["Close"].pct_change() * 100
df["Cumulative Return"] = (1 + df["Close"].pct_change()).cumprod()
df["MA7"]               = df["Close"].rolling(7).mean()
df["MA30"]              = df["Close"].rolling(30).mean()
df["MA90"]              = df["Close"].rolling(90).mean()
df["MA200"]             = df["Close"].rolling(200).mean()
df["Volatility30"]      = df["Daily Return %"].rolling(30).std()
df["Upper_BB"]          = df["MA30"] + 2 * df["Close"].rolling(30).std()
df["Lower_BB"]          = df["MA30"] - 2 * df["Close"].rolling(30).std()
df["Up_Day"]            = df["Daily Return %"] > 0

print(f"\nDate range : {df.index.min().date()}  →  {df.index.max().date()}")
print(f"Trading days: {len(df):,}")
print(f"Price range : ${df['Close'].min():.2f}  –  ${df['Close'].max():.2f}")
print(f"Total return: {(df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100:.1f}%")

# =============================================================================
# STEP 3 — TREND ANALYSIS
# =============================================================================
section("STEP 3 — TREND ANALYSIS: PRICE + MOVING AVERAGES")

fig = plt.figure(figsize=(16, 9))
gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1],
                        hspace=0.08, figure=fig)

ax1 = fig.add_subplot(gs[0])   # price + MAs
ax2 = fig.add_subplot(gs[1], sharex=ax1)   # volume
ax3 = fig.add_subplot(gs[2], sharex=ax1)   # daily returns

# ── Price & Moving Averages ───────────────────────────────────────────────────
ax1.fill_between(df.index, df["Close"], df["Close"].min() * 0.97,
                 alpha=0.07, color=UP_COLOR)
ax1.plot(df.index, df["Close"],   color=UP_COLOR,   lw=1.0,  alpha=0.9, label="Close Price")
ax1.plot(df.index, df["MA7"],     color=GOLD,        lw=1.2,  alpha=0.85, label="MA-7")
ax1.plot(df.index, df["MA30"],    color="#60A5FA",   lw=1.5,  alpha=0.85, label="MA-30")
ax1.plot(df.index, df["MA90"],    color=PURPLE,      lw=1.8,  alpha=0.85, label="MA-90")
ax1.plot(df.index, df["MA200"],   color=DOWN_COLOR,  lw=2.0,  alpha=0.90, label="MA-200 (trend)")

# Bollinger Bands
ax1.fill_between(df.index, df["Upper_BB"], df["Lower_BB"],
                 alpha=0.06, color="#60A5FA", label="Bollinger Band (±2σ)")
ax1.plot(df.index, df["Upper_BB"], color="#60A5FA", lw=0.6, linestyle="--", alpha=0.5)
ax1.plot(df.index, df["Lower_BB"], color="#60A5FA", lw=0.6, linestyle="--", alpha=0.5)

# Annotate all-time-high
ath_date  = df["Close"].idxmax()
ath_price = df["Close"].max()
ax1.annotate(f"ATH ${ath_price:.0f}",
             xy=(ath_date, ath_price),
             xytext=(60, -30), textcoords="offset points",
             arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
             fontsize=8, color="white",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD_BG,
                       edgecolor=GOLD, alpha=0.9))

ax1.set_title(f"{TICKER} — Price History with Moving Averages & Bollinger Bands",
              pad=10, color="white")
ax1.set_ylabel("Price ($)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
ax1.legend(loc="upper left", ncol=3)
ax1.tick_params(labelbottom=False)

# ── Volume Bars ───────────────────────────────────────────────────────────────
colors_vol = [UP_COLOR if u else DOWN_COLOR for u in df["Up_Day"]]
ax2.bar(df.index, df["Volume"], color=colors_vol, alpha=0.6, width=1)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax2.set_ylabel("Volume")
ax2.tick_params(labelbottom=False)

# ── Daily Returns ─────────────────────────────────────────────────────────────
ret_colors = [UP_COLOR if r > 0 else DOWN_COLOR
              for r in df["Daily Return %"].fillna(0)]
ax3.bar(df.index, df["Daily Return %"].fillna(0),
        color=ret_colors, alpha=0.7, width=1)
ax3.axhline(0, color="white", lw=0.5)
ax3.set_ylabel("Daily Ret %")
ax3.set_xlabel("Date")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.setp(ax3.get_xticklabels(), rotation=30, ha="right")

plt.suptitle(f"{TICKER} Technical Analysis Dashboard",
             fontsize=16, fontweight="bold", color="white", y=0.98)
save(fig, "01_price_trend.png")

# =============================================================================
# STEP 4 — SEASONALITY DETECTION
# =============================================================================
section("STEP 4 — SEASONALITY DETECTION")

# ── 4a. Monthly average price heatmap ─────────────────────────────────────────
monthly_pivot = (
    df["Close"]
    .resample("ME").mean()
    .to_frame()
)
monthly_pivot["Year"]  = monthly_pivot.index.year
monthly_pivot["Month"] = monthly_pivot.index.month

heatmap_data = monthly_pivot.pivot_table(
    index="Year", columns="Month", values="Close", aggfunc="mean"
)
heatmap_data.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Heatmap
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="RdYlGn",
            ax=axes[0], linewidths=0.5, linecolor=DARK_BG,
            cbar_kws={"label": "Avg Close ($)", "shrink": 0.8})
axes[0].set_title("Average Monthly Close Price Heatmap (Year × Month)")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Year")

# Average return per calendar month (seasonality bar)
df["Month"]      = df.index.month
df["Month_Name"] = df.index.strftime("%b")
month_order = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
monthly_ret = (
    df.groupby("Month_Name")["Daily Return %"]
    .mean()
    .reindex(month_order)
    .reset_index()
)
bar_colors = [UP_COLOR if v > 0 else DOWN_COLOR
              for v in monthly_ret["Daily Return %"]]
bars = axes[1].bar(monthly_ret["Month_Name"], monthly_ret["Daily Return %"],
                   color=bar_colors, edgecolor=DARK_BG, linewidth=0.6)
axes[1].axhline(0, color="white", lw=0.6)
for bar, val in zip(bars, monthly_ret["Daily Return %"]):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.001 if val >= 0 else -0.004),
                 f"{val:.3f}%", ha="center", fontsize=7.5, color=TEXT_COL)
axes[1].set_title("Average Daily Return by Calendar Month (Seasonality)")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Avg Daily Return (%)")

plt.suptitle("Seasonality Analysis", fontsize=15, fontweight="bold",
             color="white", y=1.02)
plt.tight_layout()
save(fig, "02_seasonality.png")

# ── 4b. Seasonal Decomposition ────────────────────────────────────────────────
if HAS_STATSMODELS:
    print("\nRunning seasonal decomposition (multiplicative model) …")
    monthly_close = df["Close"].resample("ME").mean().dropna()

    decomp = seasonal_decompose(monthly_close, model="multiplicative", period=12)

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    components = [
        (monthly_close,      "Observed",   UP_COLOR),
        (decomp.trend,       "Trend",      GOLD),
        (decomp.seasonal,    "Seasonal",   PURPLE),
        (decomp.resid,       "Residual",   "#60A5FA"),
    ]
    for ax, (data, label, color) in zip(axes, components):
        ax.plot(data.index, data, color=color, lw=1.6)
        if label == "Residual":
            ax.axhline(1.0,  # multiplicative residual baseline
                       color="white", lw=0.6, linestyle="--")
        ax.set_ylabel(label, color=color)
        ax.tick_params(axis="y", labelcolor=color)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")
    plt.suptitle("Seasonal Decomposition — Monthly Close Price\n"
                 "(Trend · Seasonal · Residual)",
                 fontsize=14, fontweight="bold", color="white", y=1.01)
    plt.tight_layout()
    save(fig, "03_decomposition.png")
else:
    print("  ⚠ Skipping decomposition (statsmodels not available)")

# =============================================================================
# STEP 5 — VOLATILITY & RETURNS ANALYSIS
# =============================================================================
section("STEP 5 — VOLATILITY & RETURN DISTRIBUTION")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# ── Return distribution histogram ────────────────────────────────────────────
ret_clean = df["Daily Return %"].dropna()
axes[0, 0].hist(ret_clean, bins=80, color=VOL_COLOR, edgecolor=DARK_BG,
                linewidth=0.3, density=True, alpha=0.8)
from scipy.stats import norm as sp_norm
mu_r, std_r = ret_clean.mean(), ret_clean.std()
x_r = np.linspace(ret_clean.min(), ret_clean.max(), 300)
axes[0, 0].plot(x_r, sp_norm.pdf(x_r, mu_r, std_r),
                color=GOLD, lw=2, label=f"Normal fit\nμ={mu_r:.3f}%, σ={std_r:.2f}%")
axes[0, 0].axvline(0, color="white", lw=0.8)
axes[0, 0].set_title("Daily Return Distribution")
axes[0, 0].set_xlabel("Daily Return (%)")
axes[0, 0].set_ylabel("Density")
axes[0, 0].legend()

# ── Rolling 30-day volatility ─────────────────────────────────────────────────
axes[0, 1].fill_between(df.index, df["Volatility30"],
                        alpha=0.25, color=DOWN_COLOR)
axes[0, 1].plot(df.index, df["Volatility30"],
                color=DOWN_COLOR, lw=1.3, label="30-day Rolling Vol")
axes[0, 1].axhline(df["Volatility30"].mean(), color=GOLD,
                   lw=1.2, linestyle="--", label="Mean Volatility")
axes[0, 1].set_title("Rolling 30-Day Volatility (%)")
axes[0, 1].set_xlabel("Date")
axes[0, 1].set_ylabel("Std Dev of Daily Returns")
axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[0, 1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=30, ha="right")
axes[0, 1].legend()

# ── Cumulative return ─────────────────────────────────────────────────────────
axes[1, 0].fill_between(df.index, df["Cumulative Return"],
                        1, alpha=0.12, color=UP_COLOR)
axes[1, 0].plot(df.index, df["Cumulative Return"],
                color=UP_COLOR, lw=1.6, label="Cumulative Return")
axes[1, 0].axhline(1, color="white", lw=0.7, linestyle="--")
axes[1, 0].set_title("Cumulative Return (Base = 1.0)")
axes[1, 0].set_xlabel("Date")
axes[1, 0].set_ylabel("Cumulative Return")
axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[1, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=30, ha="right")
axes[1, 0].legend()

# ── Annual returns bar chart ───────────────────────────────────────────────────
annual_ret = df["Close"].resample("YE").last().pct_change().dropna() * 100
yr_colors  = [UP_COLOR if r > 0 else DOWN_COLOR for r in annual_ret]
bars = axes[1, 1].bar(annual_ret.index.year, annual_ret.values,
                      color=yr_colors, edgecolor=DARK_BG, linewidth=0.6, width=0.6)
axes[1, 1].axhline(0, color="white", lw=0.8)
for bar, val in zip(bars, annual_ret.values):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.8 if val >= 0 else -2.5),
                    f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold",
                    color=TEXT_COL)
axes[1, 1].set_title("Annual Return (%)")
axes[1, 1].set_xlabel("Year")
axes[1, 1].set_ylabel("Annual Return (%)")
axes[1, 1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

plt.suptitle("Volatility & Return Analysis",
             fontsize=15, fontweight="bold", color="white", y=1.01)
plt.tight_layout()
save(fig, "04_volatility_returns.png")

# =============================================================================
# STEP 6 — FORECASTING (Moving Average + ARIMA)
# =============================================================================
section("STEP 6 — FORECASTING")

FORECAST_DAYS = 60

# ── Method A: Rolling-Mean Forecast (always available) ────────────────────────
last_30 = df["Close"].iloc[-30:]
ma_forecast = [last_30.mean()] * FORECAST_DAYS
forecast_dates = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1),
                                periods=FORECAST_DAYS)

# ── Method B: ARIMA (if statsmodels present) ─────────────────────────────────
arima_fc    = None
arima_ci    = None
arima_dates = forecast_dates

if HAS_STATSMODELS:
    print("\nFitting ARIMA(2,1,2) on last 500 trading days …")
    try:
        train_series = df["Close"].iloc[-500:]
        arima_model  = ARIMA(train_series, order=(2, 1, 2))
        arima_result = arima_model.fit()
        fc_obj       = arima_result.get_forecast(steps=FORECAST_DAYS)
        arima_fc     = fc_obj.predicted_mean
        arima_ci     = fc_obj.conf_int(alpha=0.10)   # 90 % CI
        arima_fc.index  = forecast_dates
        arima_ci.index  = forecast_dates
        print(f"  ARIMA AIC : {arima_result.aic:.2f}")
        print(f"  Forecast range: ${arima_fc.min():.2f} – ${arima_fc.max():.2f}")
    except Exception as e:
        print(f"  ARIMA fitting failed: {e}")

# ── Plot Forecast ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 6))

# Historical (last 180 days)
hist = df["Close"].iloc[-180:]
ax.fill_between(hist.index, hist, hist.min() * 0.97,
                alpha=0.07, color=UP_COLOR)
ax.plot(hist.index, hist, color=UP_COLOR, lw=1.5, label="Historical Close")

# MA forecast
ax.plot(forecast_dates, ma_forecast, color=GOLD, lw=1.8,
        linestyle="--", marker="o", markersize=3, label=f"MA-30 Forecast ({FORECAST_DAYS}d)")

# ARIMA forecast
if arima_fc is not None:
    ax.plot(arima_fc.index, arima_fc, color=PURPLE, lw=2.0,
            label=f"ARIMA(2,1,2) Forecast ({FORECAST_DAYS}d)")
    ax.fill_between(arima_ci.index,
                    arima_ci.iloc[:, 0], arima_ci.iloc[:, 1],
                    alpha=0.15, color=PURPLE, label="90% Confidence Interval")

# Divider line
ax.axvline(df.index[-1], color="white", lw=0.8, linestyle="--", alpha=0.6)
ax.text(df.index[-1], ax.get_ylim()[0],
        "  Forecast →", color="white", fontsize=9, va="bottom", alpha=0.7)

ax.set_title(f"{TICKER} — {FORECAST_DAYS}-Day Price Forecast", pad=10)
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
ax.legend()
plt.tight_layout()
save(fig, "05_forecast.png")

# =============================================================================
# STEP 7 — ADF STATIONARITY TEST
# =============================================================================
if HAS_STATSMODELS:
    section("STEP 7 — STATIONARITY TEST (ADF)")
    adf_result = adfuller(df["Close"].dropna())
    print(f"\n  ADF Statistic : {adf_result[0]:.4f}")
    print(f"  p-value       : {adf_result[1]:.4f}")
    print(f"  Stationary    : {'YES ✔' if adf_result[1] < 0.05 else 'NO — differencing needed'}")

    adf_diff = adfuller(df["Close"].diff().dropna())
    print(f"\n  After 1st differencing:")
    print(f"  ADF Statistic : {adf_diff[0]:.4f}")
    print(f"  p-value       : {adf_diff[1]:.4f}")
    print(f"  Stationary    : {'YES ✔' if adf_diff[1] < 0.05 else 'NO'}")

# =============================================================================
# STEP 8 — CANDLESTICK-STYLE SUMMARY CHART (last 60 days OHLC)
# =============================================================================
section("STEP 8 — OHLC SUMMARY (Last 60 Trading Days)")

recent = df.iloc[-60:].copy()

fig, axes = plt.subplots(2, 1, figsize=(15, 8),
                         gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

# Candlestick bars
for _, row in recent.iterrows():
    color = UP_COLOR if row["Close"] >= row["Open"] else DOWN_COLOR
    # High-Low wick
    axes[0].plot([row.name, row.name], [row["Low"], row["High"]],
                 color=color, lw=1.0, alpha=0.8)
    # Open-Close body
    axes[0].bar(row.name,
                abs(row["Close"] - row["Open"]),
                bottom=min(row["Open"], row["Close"]),
                color=color, width=0.6, alpha=0.85)

axes[0].plot(recent.index, recent["MA7"],  color=GOLD,   lw=1.4, label="MA-7")
axes[0].plot(recent.index, recent["MA30"], color="#60A5FA", lw=1.4, label="MA-30")
axes[0].set_title(f"{TICKER} — Candlestick Chart (Last 60 Trading Days)")
axes[0].set_ylabel("Price ($)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
axes[0].legend()

# Volume
vol_colors = [UP_COLOR if u else DOWN_COLOR for u in recent["Up_Day"]]
axes[1].bar(recent.index, recent["Volume"], color=vol_colors, alpha=0.6, width=0.7)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
axes[1].set_ylabel("Volume")
axes[1].set_xlabel("Date")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")

plt.suptitle("Recent Price Action", fontsize=14, fontweight="bold",
             color="white", y=1.01)
plt.tight_layout()
save(fig, "06_candlestick.png")

# =============================================================================
# FINAL INSIGHTS REPORT
# =============================================================================
print("\n")
print("=" * 65)
print("        TIME SERIES INSIGHTS REPORT  —  " + TICKER)
print("=" * 65)

total_ret  = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
best_day   = df["Daily Return %"].idxmax()
worst_day  = df["Daily Return %"].idxmin()
avg_vol_m  = df["Volume"].mean() / 1e6
best_year  = annual_ret.idxmax().year if len(annual_ret) > 0 else "N/A"
worst_year = annual_ret.idxmin().year if len(annual_ret) > 0 else "N/A"

trend_dir  = "UPTREND" if df["MA30"].iloc[-1] > df["MA200"].iloc[-1] else "DOWNTREND"
trend_icon = "📈" if trend_dir == "UPTREND" else "📉"

print(f"""
┌──────────────────────────────────────────────────────────┐
│  EXECUTIVE SUMMARY                                       │
├──────────────────────────────────────────────────────────┤
│  Ticker         : {TICKER:<40}│
│  Period         : {str(df.index.min().date()):<15} →  {str(df.index.max().date()):<13}│
│  Trading Days   : {len(df):<40,}│
│  Starting Price : ${df['Close'].iloc[0]:<39.2f}│
│  Ending Price   : ${df['Close'].iloc[-1]:<39.2f}│
│  Total Return   : {total_ret:<39.1f}%│
│  All-Time High  : ${df['Close'].max():<20.2f} on {str(ath_date.date()):<14}│
│  Current Trend  : {trend_icon}  {trend_dir:<36}│
└──────────────────────────────────────────────────────────┘

📈  TREND ANALYSIS
  • MA-30 vs MA-200 : {"Price ABOVE 200-day MA → Bullish long-term signal" if trend_dir=="UPTREND" else "Price BELOW 200-day MA → Bearish long-term signal"}
  • The stock has gained {total_ret:.1f}% over the full period.
  • Bollinger Bands indicate current volatility is
    {"ELEVATED — bands are wide." if df["Volatility30"].iloc[-1] > df["Volatility30"].mean() else "COMPRESSED — bands are narrow; potential breakout ahead."}

📅  SEASONALITY INSIGHTS
  • Highest avg return months: {monthly_ret.sort_values("Daily Return %").iloc[-1]["Month_Name"]} & {monthly_ret.sort_values("Daily Return %").iloc[-2]["Month_Name"]}
  • Lowest avg return months : {monthly_ret.sort_values("Daily Return %").iloc[0]["Month_Name"]} & {monthly_ret.sort_values("Daily Return %").iloc[1]["Month_Name"]}
  • Best year: {best_year} ({annual_ret.max():.1f}%)   |   Worst year: {worst_year} ({annual_ret.min():.1f}%)

⚡  VOLATILITY
  • Avg daily vol (30-day): {df["Volatility30"].mean():.2f}%  |  Current: {df["Volatility30"].iloc[-1]:.2f}%
  • Best single day  : +{df["Daily Return %"].max():.2f}%  on {str(best_day.date())}
  • Worst single day :  {df["Daily Return %"].min():.2f}%  on {str(worst_day.date())}
  • Daily return is approx. normally distributed (slight fat tails).

📊  VOLUME
  • Average daily volume : {avg_vol_m:.1f}M shares
  • Volume spikes on large-move days confirm genuine price action
    (not low-liquidity noise).

🔮  FORECAST  ({FORECAST_DAYS}-day horizon)
  • MA-30 model predicts ~${np.mean(ma_forecast):.2f} (flat mean reversion).
""" + (f"""  • ARIMA(2,1,2) forecast range: ${arima_fc.min():.2f} – ${arima_fc.max():.2f}
  • NOTE: All forecasts carry uncertainty. Use as one input
    among many, never as sole decision basis.
""" if arima_fc is not None else "  • Install statsmodels for ARIMA forecast.\n") + f"""
⚠️   KEY RISKS & OBSERVATIONS
  1. Fat-tail events: the worst day exceeded -5% — tail risk is real.
  2. Volatility clusters: high-vol regimes can persist for weeks.
  3. Past trends do NOT guarantee future performance.
  4. Always combine technical signals with fundamental research.

📁  Output charts saved to: ./{OUTPUT_DIR}/
""")
print("=" * 65)
print("Analysis complete ✔")


from PIL import Image
img=Image.open("images/stock_timeseries_dashboard.png")
img.show()