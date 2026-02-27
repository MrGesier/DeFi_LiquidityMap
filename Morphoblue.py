import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import hashlib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Vault Zones + Market Health Monitor", layout="wide")
st.title("🧭 Vault Operational Zones + Market Health Monitor")
st.caption("Unified tool — Zones de fonctionnement des vaults + Liquidation feasibility + Health score | by Mr_Gesier")

API_POOLS = "https://yields.llama.fi/pools"
API_CHART = "https://yields.llama.fi/chart/"
MAX_VAULTS_DEFAULT = 80

# =========================================================
# HELPERS
# =========================================================
def reproducible_fallback_ratio(key: str, low=0.30, high=0.90) -> float:
    """Fallback déterministe quand une série/API manque (évite un random qui change à chaque refresh)."""
    h = hashlib.sha256(key.encode()).hexdigest()
    v = int(h[:8], 16) / 0xffffffff
    return low + (high - low) * v

def clamp01(x):
    if pd.isna(x):
        return np.nan
    return float(np.clip(x, 0.0, 1.0))

# =========================================================
# FETCH DATA — DEFI LLAMA
# =========================================================
@st.cache_data(show_spinner=True, ttl=3600)
def fetch_pools(limit=MAX_VAULTS_DEFAULT) -> pd.DataFrame:
    """
    Pull snapshot of top TVL pools from DeFiLlama yields endpoint.
    """
    r = requests.get(API_POOLS, timeout=20)
    r.raise_for_status()
    data = r.json().get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df

    keep = ["pool", "project", "chain", "symbol", "tvlUsd", "apy", "apyMean30d", "stablecoin", "ilRisk"]
    df = df[keep].copy()

    df["apy"] = pd.to_numeric(df["apy"], errors="coerce") / 100.0
    df["apyMean30d"] = pd.to_numeric(df["apyMean30d"], errors="coerce") / 100.0
    df["tvlUsd"] = pd.to_numeric(df["tvlUsd"], errors="coerce")

    df = df.dropna(subset=["apy", "tvlUsd"])
    df = df.sort_values("tvlUsd", ascending=False).head(int(limit))
    df["name"] = df.apply(lambda r: f"{r['project']}:{r['symbol']} ({r['chain']})", axis=1)
    return df.reset_index(drop=True)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_history(pool_id: str):
    """
    DeFiLlama chart endpoint (per pool).
    - Often includes APY series.
    - Sometimes includes tvlUsd series (not guaranteed).
    """
    url = f"{API_CHART}{pool_id}"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200 or not r.text.strip().startswith("{"):
            return None
        js = r.json()
        if "data" not in js or len(js["data"]) == 0:
            return None

        df = pd.DataFrame(js["data"])
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        for col in ["apy", "apyBase", "apyReward", "tvlUsd"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "apy" in df.columns:
            df["apy"] = df["apy"] / 100.0
        return df
    except Exception:
        return None

@st.cache_data(show_spinner=True, ttl=3600)
def build_hybrid_metrics(snapshot: pd.DataFrame):
    """
    Builds mean_return + volatility per vault using APY history if available,
    else fallback to snapshot APY and APYMean30d diff.
    """
    rows, skipped = [], []
    progress = st.progress(0.0)
    n = len(snapshot)

    for i, row in snapshot.iterrows():
        pid = row["pool"]
        hist = fetch_history(pid)

        if hist is not None and "apy" in hist.columns and hist["apy"].notna().sum() >= 5:
            mean_ret = float(hist["apy"].dropna().mean())
            vol = float(hist["apy"].dropna().std())
        else:
            mean_ret = float(row["apy"])
            vol = float(abs(row["apy"] - row["apyMean30d"])) if not pd.isna(row["apyMean30d"]) else 0.01
            skipped.append(pid)

        rows.append({
            "pool": pid,
            "name": row["name"],
            "project": row["project"],
            "chain": row["chain"],
            "symbol": row["symbol"],
            "tvlUsd": float(row["tvlUsd"]),
            "stablecoin": bool(row["stablecoin"]),
            "ilRisk": row["ilRisk"],
            "mean_return": max(0.0, mean_ret),
            "volatility": max(0.0001, vol),
        })

        progress.progress((i + 1) / max(1, n))
        time.sleep(0.003)

    return pd.DataFrame(rows), skipped

# =========================================================
# CLASSIFICATION
# =========================================================
def classify(row):
    if row["stablecoin"]:
        return "Stablecoin"
    s = str(row["symbol"]).upper()
    if "BTC" in s:
        return "BTC"
    if any(k in s for k in ["ETH", "STETH", "WSTETH", "WBETH", "RETH", "CBETH"]):
        return "ETH / LSD"
    if str(row["ilRisk"]).lower() == "yes":
        return "LP / Farming"
    return "Structured / Credit"

COLOR_MAP = {
    "Stablecoin": "#2E6BE6",
    "BTC": "#F59E0B",
    "ETH / LSD": "#10B981",
    "LP / Farming": "#FBBF24",
    "Structured / Credit": "#8B5CF6",
}

# =========================================================
# LIVE USAGE + CAPACITY (BEST-EFFORT)
# =========================================================
@st.cache_data(ttl=3600)
def try_fetch_usage_ratios() -> dict:
    """
    Protocol-level utilization ratios (best effort).
    Returns dict {protocol_key: ratio_in_[0..1]}.
    """
    usage = {}

    # --- Aave v3/v2 ---
    vals = []
    for url in ["https://aave-api-v3.aave.com/data/reserves", "https://aave-api-v2.aave.com/data/reserves"]:
        try:
            j = requests.get(url, timeout=10).json()
            for x in j:
                for k in ["utilizationRate", "liquidityUsageRatio", "borrowUsageRatio"]:
                    if k in x:
                        vals.append(pd.to_numeric(x[k], errors="coerce"))
        except Exception:
            continue
    usage["aave"] = float(np.clip(np.nanmean(vals), 0, 1)) if len(vals) else np.nan

    # --- Compound v2 ---
    try:
        j = requests.get("https://api.compound.finance/api/v2/ctoken", timeout=10).json()
        vals = []
        for c in j.get("cToken", []):
            tb = pd.to_numeric(c.get("total_borrows", {}).get("value", None), errors="coerce")
            ts = pd.to_numeric(c.get("total_supply", {}).get("value", None), errors="coerce")
            if pd.notna(tb) and pd.notna(ts) and ts > 0:
                vals.append(tb / ts)
        usage["compound"] = float(np.clip(np.nanmean(vals), 0, 1)) if len(vals) else np.nan
    except Exception:
        usage["compound"] = np.nan

    # --- Morpho (endpoint may vary) ---
    try:
        j = requests.get("https://api.morpho.xyz/morpho-markets", timeout=10).json()
        vals = [pd.to_numeric(m.get("utilization", None), errors="coerce") for m in j]
        vals = pd.Series(vals).dropna()
        usage["morpho"] = float(np.clip(vals.mean(), 0, 1)) if len(vals) else np.nan
    except Exception:
        usage["morpho"] = np.nan

    # --- Gearbox ---
    try:
        j = requests.get("https://api.gearbox.fi/v3/pools", timeout=10).json()
        vals = [pd.to_numeric(p.get("utilizationRate", None), errors="coerce") for p in j]
        vals = pd.Series(vals).dropna()
        usage["gearbox"] = float(np.clip(vals.mean(), 0, 1)) if len(vals) else np.nan
    except Exception:
        usage["gearbox"] = np.nan

    return usage

@st.cache_data(ttl=3600)
def enrich_live_risks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - capacity_ratio: current TVL / historical max TVL (if tvlUsd series exists)
      - usage_ratio: protocol utilization ratio (Aave/Compound/Morpho/Gearbox) via substring match on project name
    """
    usage_data = try_fetch_usage_ratios()
    cap_ratios, use_ratios = [], []

    for _, r in df.iterrows():
        hist = fetch_history(r["pool"])
        cap_ratio = np.nan
        if hist is not None and "tvlUsd" in hist.columns:
            tvl_hist = hist["tvlUsd"].dropna()
            if len(tvl_hist) >= 5 and tvl_hist.max() > 0:
                cap_ratio = float(r["tvlUsd"]) / float(tvl_hist.max())

        prj = str(r["project"]).lower()
        usage_ratio = np.nan
        for key in usage_data.keys():
            if key in prj:
                usage_ratio = usage_data[key]
                break

        cap_ratios.append(cap_ratio)
        use_ratios.append(usage_ratio)

    out = df.copy()
    out["capacity_ratio"] = cap_ratios
    out["usage_ratio"] = use_ratios
    return out

# =========================================================
# VAULT ZONES
# =========================================================
def compute_vault_zone(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Zones on (L=usage_ratio, K=capacity_ratio), both projected into [0..1] for the zone checks.
    If missing, we fill with deterministic fallback (flag still accessible via raw columns).
    """
    out = df.copy()

    LminV, LmaxV = params["L_min_vault"], params["L_max_vault"]
    KminV, KmaxV = params["K_min_vault"], params["K_max_vault"]
    LminO, LmaxO = params["L_min_opt"], params["L_max_opt"]
    KminO, KmaxO = params["K_min_opt"], params["K_max_opt"]

    L_used, K_used = [], []
    for _, r in out.iterrows():
        lv = r.get("usage_ratio", np.nan)
        kv = r.get("capacity_ratio", np.nan)

        if pd.isna(lv):
            lv = reproducible_fallback_ratio(r["pool"] + r["name"], low=0.35, high=0.85)
        if pd.isna(kv):
            kv = reproducible_fallback_ratio(r["name"] + r["pool"], low=0.45, high=0.80)

        L_used.append(float(np.clip(lv, 0.0, 1.0)))
        K_used.append(float(np.clip(kv, 0.0, 1.0)))

    out["L_used"] = L_used
    out["K_used"] = K_used
    out["K_raw"] = out["capacity_ratio"]

    in_vault = (
        (out["L_used"] >= LminV) & (out["L_used"] <= LmaxV) &
        (out["K_used"] >= KminV) & (out["K_used"] <= KmaxV)
    )
    in_opt = (
        (out["L_used"] >= LminO) & (out["L_used"] <= LmaxO) &
        (out["K_used"] >= KminO) & (out["K_used"] <= KmaxO)
    )

    out["zone"] = np.where(in_opt, "Optimal", np.where(in_vault, "Acceptable", "Out of Spec"))
    out["zone_emoji"] = out["zone"].map({"Optimal": "🟢", "Acceptable": "🟡", "Out of Spec": "🔴"})
    return out

def plot_vault_zone_scatter(dfz: pd.DataFrame, params: dict, selected_name: str | None):
    fig = px.scatter(
        dfz,
        x="L_used",
        y="K_used",
        color="zone",
        size="tvlUsd",
        hover_data={
            "name": True,
            "project": True,
            "chain": True,
            "symbol": True,
            "tvlUsd": ":,.0f",
            "usage_ratio": ":.3f",
            "capacity_ratio": ":.3f",
            "L_used": ":.3f",
            "K_used": ":.3f",
            "zone": True
        },
        category_orders={"zone": ["Optimal", "Acceptable", "Out of Spec"]},
        color_discrete_map={"Optimal": "#10B981", "Acceptable": "#F59E0B", "Out of Spec": "#EF4444"},
        labels={"L_used": "Loan usage ratio (utilization)", "K_used": "Capacity utilization (normalized)"},
        title="Vault Operational Zones — Universe View",
    )

    # Vault limits
    fig.add_shape(
        type="rect",
        x0=params["L_min_vault"], x1=params["L_max_vault"],
        y0=params["K_min_vault"], y1=params["K_max_vault"],
        line=dict(width=2, color="#EF4444", dash="dash"),
        fillcolor="rgba(239, 68, 68, 0.06)",
        layer="below"
    )
    fig.add_annotation(
        x=params["L_min_vault"], y=params["K_max_vault"],
        text="Vault Limits", showarrow=False, yshift=10, xshift=10
    )

    # Optimal zone
    fig.add_shape(
        type="rect",
        x0=params["L_min_opt"], x1=params["L_max_opt"],
        y0=params["K_min_opt"], y1=params["K_max_opt"],
        line=dict(width=2, color="#FBBF24", dash="dot"),
        fillcolor="rgba(251, 191, 36, 0.10)",
        layer="below"
    )
    fig.add_annotation(
        x=params["L_min_opt"], y=params["K_max_opt"],
        text="Optimal Zone", showarrow=False, yshift=10, xshift=10
    )

    if selected_name:
        sel = dfz[dfz["name"] == selected_name]
        if not sel.empty:
            fig.add_trace(
                go.Scatter(
                    x=sel["L_used"],
                    y=sel["K_used"],
                    mode="markers+text",
                    marker=dict(size=16, symbol="x", color="black"),
                    text=["Selected"],
                    textposition="top center",
                    name="Selected vault",
                    hoverinfo="skip",
                )
            )

    fig.update_layout(height=620, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig

# =========================================================
# LIQUIDATION FEASIBILITY (SECOND TOOL) — PACKAGED
# =========================================================
def lif_from_lltv(lltv_val: float) -> float:
    return max(0.0, (1.0 - lltv_val) / max(1e-9, lltv_val))

def effective_loss(beta: np.ndarray, delta: np.ndarray, base_fee: float, conv: float, amp: float) -> np.ndarray:
    slip0 = base_fee + (conv * (beta**2) / (1.0 - beta))
    mult = 1.0 + amp * (delta**2)
    return slip0 * mult

def dislocation_boundary(beta: np.ndarray, mid: float, steep: float, base: float, span: float) -> np.ndarray:
    return base + span * (1.0 / (1.0 + np.exp(-(beta - mid) * steep)))

def build_liquidation_map(params: dict):
    """
    Returns:
      fig, verdict_str, zone_code_current, metrics_dict
    zone_current:
      0 safe
      1 marginal
      2 econ-unsafe
      3 dislocation
    """
    x_mode = params["x_mode"]
    market_depth_usd = params["market_depth_usd"]
    haircut_pct = params["haircut_pct"]
    basis_pct = params["basis_pct"]
    w_haircut = params["w_haircut"]
    w_basis = params["w_basis"]
    lltv = params["lltv"]
    lif_model = params["lif_model"]
    base_fee_pct = params["base_fee_pct"]
    convexity = params["convexity"]
    stress_amp = params["stress_amp"]
    disloc_mid = params["disloc_mid"]
    disloc_steep = params["disloc_steep"]
    disloc_base = params["disloc_base"]
    disloc_span = params["disloc_span"]
    beta_current = float(np.clip(params["beta_current"], 0.01, 0.99))

    # Δ (stress index)
    delta_current = (w_haircut * (haircut_pct / 100.0) * 10.0) + (w_basis * (basis_pct / 10.0))
    effective_depth_usd = market_depth_usd * (1.0 - haircut_pct / 100.0)
    liquidation_size_usd = effective_depth_usd * beta_current

    # LIF
    lif_proxy = lif_from_lltv(lltv)
    lif_5 = 0.05
    lif_10 = 0.10

    if lif_model.startswith("Proxy"):
        lif_lines = [(lif_proxy, f"LIF from LLTV ({lif_proxy*100:.1f}%)")]
        lif_main = lif_proxy
    else:
        lif_lines = [(lif_5, "LIF 5%"), (lif_10, "LIF 10%")]
        lif_main = lif_10

    # Grid
    beta_grid = np.linspace(0.01, 0.99, 280)
    delta_grid = np.linspace(0.0, 10.0, 280)
    B, D = np.meshgrid(beta_grid, delta_grid)

    loss_grid = effective_loss(B, D, base_fee_pct / 100.0, convexity, stress_amp)
    disloc_grid = dislocation_boundary(B, disloc_mid, disloc_steep, disloc_base, disloc_span)

    is_disloc = D >= disloc_grid
    unsafe_econ = loss_grid > lif_main

    zone = np.zeros_like(loss_grid, dtype=int)
    zone[(unsafe_econ) & (~is_disloc)] = 2
    zone[is_disloc] = 3

    if not lif_model.startswith("Proxy"):
        zone[(loss_grid > lif_5) & (loss_grid <= lif_10) & (~is_disloc)] = 1
        zone[(loss_grid <= lif_5) & (~is_disloc)] = 0

    # X-axis
    if x_mode == "$ (absolute)":
        x_grid = beta_grid * effective_depth_usd
        x_current = float(liquidation_size_usd)
        x_title = "Liquidation size / executable depth ($)"
    else:
        x_grid = beta_grid * 100.0
        x_current = float(beta_current * 100.0)
        x_title = "Liquidation size / executable depth (%)"

    # Current scenario
    loss_current = float(effective_loss(np.array([beta_current]), np.array([delta_current]),
                                        base_fee_pct / 100.0, convexity, stress_amp)[0])
    disloc_current = float(dislocation_boundary(np.array([beta_current]), disloc_mid, disloc_steep,
                                               disloc_base, disloc_span)[0])

    if delta_current >= disloc_current:
        zone_current = 3
        verdict = "🛑 Dislocation regime: liquidity collapse / execution failure likely."
    elif loss_current > lif_main:
        zone_current = 2
        verdict = "⚠️ Econ-unsafe: effective loss > incentive → liquidators may abstain → bad debt risk."
    else:
        if not lif_model.startswith("Proxy"):
            if (loss_current > lif_5) and (loss_current <= lif_10) and (delta_current < disloc_current):
                zone_current = 1
                verdict = "🟡 Marginal: sensitive to incentives & assumptions."
            else:
                zone_current = 0
                verdict = "✅ Economically feasible: incentive covers stressed execution loss."
        else:
            zone_current = 0
            verdict = "✅ Economically feasible: incentive covers stressed execution loss."

    # Plotly
    fig = go.Figure()

    colorscale = [
        [0.00, "#C8F7C5"], [0.2499, "#C8F7C5"],  # safe
        [0.25, "#FFF3B0"], [0.4999, "#FFF3B0"],  # marginal
        [0.50, "#FFD1D1"], [0.7499, "#FFD1D1"],  # econ-unsafe
        [0.75, "#D6E4FF"], [1.00, "#D6E4FF"],    # dislocation
    ]

    fig.add_trace(
        go.Heatmap(
            x=x_grid,
            y=delta_grid,
            z=zone,
            zmin=0,
            zmax=3,
            colorscale=colorscale,
            opacity=0.55,
            colorbar=dict(
                title="Zone",
                tickmode="array",
                tickvals=[0, 1, 2, 3],
                ticktext=["Safe", "Marginal", "Econ-unsafe", "Dislocation"],
            ),
            hovertemplate=(
                ("Liquidation size: $%{x:,.0f}<br>" if x_mode == "$ (absolute)" else "Liquidation size: %{x:.1f}%<br>")
                + "Δ (stress): %{y:.2f}<br>"
                + "Zone code: %{z}<extra></extra>"
            ),
        )
    )

    for lvl, lbl in lif_lines:
        fig.add_trace(
            go.Contour(
                x=x_grid,
                y=delta_grid,
                z=loss_grid,
                contours=dict(start=lvl, end=lvl, size=1e-6, coloring="none"),
                line=dict(color="red", width=3),
                showscale=False,
                name=lbl,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=dislocation_boundary(beta_grid, disloc_mid, disloc_steep, disloc_base, disloc_span),
            mode="lines",
            line=dict(color="black", width=3),
            name="Dislocation boundary",
            hovertemplate=(
                ("Liquidation size: $%{x:,.0f}<br>" if x_mode == "$ (absolute)" else "Liquidation size: %{x:.1f}%<br>")
                + "Dislocation Δ: %{y:.2f}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[x_current],
            y=[delta_current],
            mode="markers",
            marker=dict(size=14, symbol="x", color="black"),
            name="Current scenario",
            hovertemplate=(
                (f"Liquidation size: ${liquidation_size_usd:,.0f}<br>" if x_mode == "$ (absolute)" else f"Liquidation size: {beta_current*100:.1f}%<br>")
                + f"Δ: {delta_current:.2f}<br>"
                + f"Effective loss: {loss_current*100:.1f}%<br>"
                + (f"LIF (proxy): {lif_proxy*100:.1f}%<extra></extra>" if lif_model.startswith("Proxy") else "LIF bands: 5% / 10%<extra></extra>")
            ),
        )
    )

    fig.update_layout(
        height=720,
        title="Liquidation Feasibility Map (zones, incentives & dislocation)",
        xaxis_title=x_title,
        yaxis_title="Δ = liquidity haircut + oracle/spot basis (stress index)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fig.add_annotation(
        x=0.01, y=0.99, xref="paper", yref="paper",
        text="<b>Rule:</b> if <b>effective loss</b> crosses <b>red LIF</b>, liquidation becomes irrational → bad debt risk.",
        showarrow=False,
        align="left",
    )

    metrics = {
        "effective_depth_usd": float(effective_depth_usd),
        "liquidation_size_usd": float(liquidation_size_usd),
        "delta_current": float(delta_current),
        "lif_proxy": float(lif_proxy),
        "loss_current": float(loss_current),
        "disloc_current": float(disloc_current),
    }

    return fig, verdict, int(zone_current), metrics

# =========================================================
# MARKET HEALTH SCORE
# =========================================================
def health_from_zones(p_opt, p_acc, p_out):
    """Health component based on zone distribution (0..1)."""
    return float(np.clip(0.65 * p_opt + 0.30 * p_acc + 0.05 * (1 - p_out), 0.0, 1.0))

def health_from_liq_zone(zone_code: int):
    """Map liquidation zone (0..3) to health (0..1)."""
    return {0: 1.0, 1: 0.55, 2: 0.20, 3: 0.0}.get(int(zone_code), 0.4)

def verdict_from_score(score_0_100: float, liq_zone: int):
    if liq_zone == 3:
        return "🛑 Dislocated"
    if score_0_100 >= 75:
        return "✅ Healthy"
    if score_0_100 >= 55:
        return "🟡 Watch"
    return "⚠️ Risky"

def plot_gauge(score):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(score),
            title={"text": "Market Health Score (0–100)"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 55], "color": "rgba(239,68,68,0.25)"},
                    {"range": [55, 75], "color": "rgba(245,158,11,0.25)"},
                    {"range": [75, 100], "color": "rgba(16,185,129,0.25)"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": float(score)},
            },
        )
    )
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=60, b=20))
    return fig

# =========================================================
# Oracle Analysis – UI inspirée des captures
# =========================================================
def generate_oracle_data(token, days):
    """
    Génère des séries temporelles simulées pour spot et TWAP.
    """
    # Paramètres spécifiques au token
    config = {
        "SOL": {"price": 150.0, "vol": 0.02},
        "BTC": {"price": 50000.0, "vol": 0.01},
        "ETH": {"price": 3000.0, "vol": 0.015},
        "USDC": {"price": 1.0, "vol": 0.0005},
        "wstETH": {"price": 3500.0, "vol": 0.015},
        "WBTC": {"price": 50000.0, "vol": 0.01},
        "LDO": {"price": 2.5, "vol": 0.03},
        "ARB": {"price": 1.2, "vol": 0.025},
    }
    cfg = config.get(token, {"price": 100.0, "vol": 0.01})

    # Création des timestamps (toutes les 5 minutes)
    end = datetime.now()
    start = end - pd.Timedelta(days=days)
    timestamps = pd.date_range(start=start, end=end, freq='5min', inclusive='left')
    n = len(timestamps)

    # Random walk pour spot
    np.random.seed(abs(hash(token)) % 2**32)
    returns = np.random.normal(0, cfg["vol"], n)
    spot = cfg["price"] * np.exp(np.cumsum(returns))

    # TWAP sur fenêtre de 12 points (1 heure)
    window = 12
    twap = pd.Series(spot).rolling(window=window, min_periods=1).mean().values

    divergence = (spot - twap) / twap * 100

    df = pd.DataFrame({
        "ts": timestamps,
        "spot_price": spot.round(4),
        "TWAP_price": twap.round(4),
        "twap_divergence_period": divergence.round(4)
    })
    return df

# =========================================================
# Loans mock data
# =========================================================
def generate_mock_loans(n=50):
    np.random.seed(123)
    owners = [f"Owner_{i}" for i in range(n)]
    pubkeys = [f"PubKey_{i}" for i in range(n)]
    elevation = np.random.choice([0, 1], n)
    current = np.random.uniform(0.7, 0.95, n)
    max_ltv = np.random.uniform(0.75, 0.92, n)
    unhealthy = (current > max_ltv * 0.98).astype(int)
    dist_to_liqu = max_ltv - current
    total_debt = np.random.uniform(0, 40, n)
    total_borrow = total_debt * np.random.uniform(0.8, 1.2, n)
    net_value = np.random.uniform(0, 10, n)
    df = pd.DataFrame({
        "owner": owners,
        "pubkey": pubkeys,
        "elevation": elevation,
        "current_ltv": current,
        "max_ltv": max_ltv,
        "unhealthy": unhealthy,
        "dist_to_liqu": dist_to_liqu,
        "total_debt": total_debt,
        "total_borrow": total_borrow,
        "net_value": net_value,
        "url": ["Link"] * n
    })
    return df

# =========================================================
# SIDEBAR — CONTROLS
# =========================================================
with st.sidebar:
    st.header("📥 Data loading")

    max_vaults = st.slider(
        "Max vaults (top TVL)",
        20, 200, MAX_VAULTS_DEFAULT, 5,
        help="Nombre de vaults/pools chargés depuis DeFiLlama, triés par TVL décroissant."
    )

    if st.button("🔄 Force refresh cache"):
        st.cache_data.clear()
        st.success("Cache cleared — Streamlit va rerun automatiquement.")

    st.markdown("---")
    st.header("🎯 Vault Operational Zones")
    st.caption("Zones sur le plan (L=usage ratio, K=capacity utilization).")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Vault Limits (Creator)**")
        L_min_vault = st.slider("Min Loan Usage (L)", 0.0, 0.5, 0.10, 0.01)
        L_max_vault = st.slider("Max Loan Usage (L)", 0.5, 1.0, 0.85, 0.01)
        K_min_vault = st.slider("Min Capacity (K)", 0.0, 0.5, 0.20, 0.01)
        K_max_vault = st.slider("Max Capacity (K)", 0.5, 1.0, 0.95, 0.01)
    with col2:
        st.markdown("**Optimal Zone (Protocol)**")
        L_min_opt = st.slider("Min Loan Usage (opt)", 0.0, 0.5, 0.25, 0.01)
        L_max_opt = st.slider("Max Loan Usage (opt)", 0.5, 1.0, 0.55, 0.01)
        K_min_opt = st.slider("Min Capacity (opt)", 0.0, 0.5, 0.35, 0.01)
        K_max_opt = st.slider("Max Capacity (opt)", 0.5, 1.0, 0.70, 0.01)

    zones_params = dict(
        L_min_vault=L_min_vault, L_max_vault=L_max_vault,
        K_min_vault=K_min_vault, K_max_vault=K_max_vault,
        L_min_opt=L_min_opt, L_max_opt=L_max_opt,
        K_min_opt=K_min_opt, K_max_opt=K_max_opt
    )

    st.markdown("---")
    st.header("🧪 Price Shock & Liquidation Feasibility")

    x_mode = st.radio("X-axis mode", ["% of executable depth", "$ (absolute)"])
    market_depth_usd = st.number_input("Executable market depth ($)", 1_000_000, 50_000_000_000, 150_000_000, step=10_000_000)
    haircut_pct = st.slider("Liquidity haircut (%)", 0, 95, 40)
    basis_pct = st.slider("Oracle ↔ spot basis (%)", 0.0, 15.0, 2.0, step=0.5)
    price_shock_pct = st.slider("Collateral price shock (%)", -50.0, 50.0, 0.0, step=1.0,
                                 help="Negative = price drop. This adds to the oracle/spot basis.")
    effective_basis = float(basis_pct + max(0, -price_shock_pct))

    w_haircut = st.slider("Weight: haircut", 0.0, 2.0, 1.0, step=0.05)
    w_basis = st.slider("Weight: basis", 0.0, 2.0, 1.0, step=0.05)

    lltv = st.slider("LLTV (Liquidation Loan-To-Value)", 0.50, 0.95, 0.86, step=0.01)
    lif_model = st.selectbox("LIF model", ["Proxy ( (1-LLTV)/LLTV )", "Fixed bands (5% / 10%)"])

    base_fee_pct = st.slider("Base execution cost (%)", 0.0, 5.0, 1.0, step=0.1)
    convexity = st.slider("Slippage convexity", 0.05, 1.00, 0.25, step=0.01)
    stress_amp = st.slider("Stress amplification (non-linear)", 0.00, 0.20, 0.06, step=0.01)

    disloc_mid = st.slider("Dislocation midpoint (β)", 0.30, 0.90, 0.60, step=0.01)
    disloc_steep = st.slider("Dislocation steepness", 5, 50, 22, step=1)
    disloc_base = st.slider("Base Δ", 0.5, 5.0, 1.8, step=0.1)
    disloc_span = st.slider("Additional Δ tolerance at high β", 0.5, 6.0, 3.8, step=0.1)

    if x_mode == "% of executable depth":
        beta_current = st.slider("Current liquidation size (% of executable depth)", 1, 100, 50) / 100.0
    else:
        beta_current = st.slider("Current β = liquidation size / depth", 0.01, 0.99, 0.50, step=0.01)

    liq_params = dict(
        x_mode=x_mode,
        market_depth_usd=float(market_depth_usd),
        haircut_pct=float(haircut_pct),
        basis_pct=effective_basis,
        w_haircut=float(w_haircut),
        w_basis=float(w_basis),
        lltv=float(lltv),
        lif_model=lif_model,
        base_fee_pct=float(base_fee_pct),
        convexity=float(convexity),
        stress_amp=float(stress_amp),
        disloc_mid=float(disloc_mid),
        disloc_steep=float(disloc_steep),
        disloc_base=float(disloc_base),
        disloc_span=float(disloc_span),
        beta_current=float(beta_current),
        price_shock_pct=price_shock_pct,
    )

# =========================================================
# LOAD & PREP DATA
# =========================================================
st.info("⏳ Loading pools from DeFiLlama…")
snap = fetch_pools(max_vaults)
if snap is None or snap.empty:
    st.error("No pool data returned from DeFiLlama.")
    st.stop()

st.info("⏳ Building per-vault metrics (APY history fallback)…")
data, skipped_ids = build_hybrid_metrics(snap)

data["category"] = data.apply(classify, axis=1)
data["color"] = data["category"].map(COLOR_MAP)
data["mean_return"] = np.clip(data["mean_return"], 0, 0.50)
data["volatility"] = np.clip(data["volatility"], 0, 0.30)

st.info("🔎 Enriching with live risk inputs (capacity_ratio + usage_ratio)…")
data = enrich_live_risks(data)

st.success(f"Loaded {len(data)} vaults. {len(data)-len(skipped_ids)} have APY history; others use snapshot fallback.")

with st.expander("ℹ️ Data caveats (important)"):
    st.write(
        "- **capacity_ratio**: proxy = TVL / max(TVL historique) si la série tvlUsd est fournie par DeFiLlama chart (sinon fallback).\n"
        "- **usage_ratio**: best-effort au niveau **protocole** (Aave/Compound/Morpho/Gearbox) via mapping par nom de projet.\n"
        "- Pour un vrai *control chart* L/K par vault, upload tes séries time-series internes."
    )

# =========================================================
# FILTERS + SELECT
# =========================================================
st.markdown("---")
cA, cB, cC = st.columns([1.2, 1.2, 1.0])

with cA:
    cats = sorted(data["category"].unique().tolist())
    selected_cats = st.multiselect("Filter by category", cats, default=cats)

with cB:
    chains = sorted(data["chain"].dropna().astype(str).unique().tolist())
    selected_chains = st.multiselect("Filter by chain", chains, default=chains)

with cC:
    min_tvl = st.number_input("Min TVL (USD)", min_value=0, value=0, step=1_000_000)

uni = data[
    (data["category"].isin(selected_cats)) &
    (data["chain"].astype(str).isin(selected_chains)) &
    (data["tvlUsd"] >= float(min_tvl))
].copy()

if uni.empty:
    st.warning("Universe empty after filters.")
    st.stop()

vault_names = uni["name"].tolist()
selected_vault = st.selectbox("Select vault (for drilldown)", vault_names, index=0)

# =========================================================
# COMPUTE ZONES
# =========================================================
uni_z = compute_vault_zone(uni, zones_params)

p_opt = float((uni_z["zone"] == "Optimal").mean())
p_acc = float((uni_z["zone"] == "Acceptable").mean())
p_out = float((uni_z["zone"] == "Out of Spec").mean())

# =========================================================
# LIQ MAP + HEALTH
# =========================================================
liq_fig, liq_verdict, liq_zone_current, liq_metrics = build_liquidation_map(liq_params)

vault_health = health_from_zones(p_opt, p_acc, p_out)
liq_health = health_from_liq_zone(liq_zone_current)

mu = uni_z["mean_return"].clip(0, 0.50).astype(float)
sig = uni_z["volatility"].clip(0.0001, 0.30).astype(float)
quality = (mu / (sig + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
q_norm = float(np.clip((quality.mean() / max(1e-9, quality.quantile(0.90))), 0.0, 1.0))

score = 100.0 * (0.55 * vault_health + 0.30 * liq_health + 0.15 * q_norm)
verdict = verdict_from_score(score, liq_zone_current)

# =========================================================
# TABS (6 tabs)
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Vault Zones (Universe)",
    "🔍 Vault Drilldown",
    "📈 Price Shock & Liquidation",
    "🧠 Market Health Monitor",
    "🔮 Oracle Analysis",
    "🏦 Loans Analysis"
])

# -----------------------------
# TAB 1 — Universe view
# -----------------------------
with tab1:
    st.subheader("Vault Operational Zones — Universe")
    st.caption("Chaque point = un vault. Axes: **L** (utilization/usage) et **K** (capacity proxy).")
    st.plotly_chart(plot_vault_zone_scatter(uni_z, zones_params, selected_vault), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 Optimal", f"{p_opt*100:.1f}%")
    c2.metric("🟡 Acceptable", f"{p_acc*100:.1f}%")
    c3.metric("🔴 Out of Spec", f"{p_out*100:.1f}%")
    c4.metric("Universe size", f"{len(uni_z)}")

    with st.expander("How to interpret vault zones"):
        st.write("- **Optimal**: dans la plage cible.\n- **Acceptable**: OK mais à surveiller.\n- **Out of Spec**: en dehors des limites → risque.")

    show_cols = [
        "zone_emoji", "zone", "name", "project", "chain", "category",
        "tvlUsd", "mean_return", "volatility",
        "usage_ratio", "capacity_ratio", "L_used", "K_used"
    ]
    st.dataframe(uni_z[show_cols].sort_values(["zone", "tvlUsd"], ascending=[True, False]), use_container_width=True, hide_index=True)

    csv = uni_z[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download vault zones CSV", csv, "vault_zones.csv", "text/csv")

# -----------------------------
# TAB 2 — Drilldown + upload
# -----------------------------
with tab2:
    st.subheader("Vault Drilldown")
    sel = uni_z[uni_z["name"] == selected_vault].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zone", f"{sel['zone_emoji']} {sel['zone']}")
    c2.metric("Usage (L)", f"{sel['L_used']:.1%}")
    c3.metric("Capacity (K)", f"{sel['K_used']:.1%}")
    c4.metric("TVL", f"${sel['tvlUsd']:,.0f}")

    st.markdown("### Capacity history (real, if available)")
    hist = fetch_history(sel["pool"])
    if hist is not None and "timestamp" in hist.columns and "tvlUsd" in hist.columns:
        h = hist.dropna(subset=["timestamp", "tvlUsd"]).sort_values("timestamp")
        if len(h) >= 5:
            max_tvl = float(h["tvlUsd"].max())
            h["capacity_ratio_hist"] = (h["tvlUsd"] / max(1e-9, max_tvl)).clip(0, 1)
            fig_cap = px.line(h, x="timestamp", y="capacity_ratio_hist",
                              title="Capacity ratio history (TVL / max(TVL over history))")
            fig_cap.update_layout(height=360)
            st.plotly_chart(fig_cap, use_container_width=True)
        else:
            st.info("Not enough TVL points in history.")
    else:
        st.info("No tvlUsd series returned for this vault.")

    st.markdown("---")
    st.markdown("### (Optional) Upload your own L/K time series for a true control chart")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            dfu = pd.read_csv(up)
            cols = {c.lower().strip(): c for c in dfu.columns}
            tcol = cols.get("timestamp")
            lcol = cols.get("l") or cols.get("usage") or cols.get("usage_ratio")
            kcol = cols.get("k") or cols.get("capacity") or cols.get("capacity_ratio")

            if not (tcol and lcol and kcol):
                st.error("CSV must include: timestamp + L + K (or usage_ratio/capacity_ratio equivalents).")
            else:
                dff = dfu[[tcol, lcol, kcol]].copy()
                dff.columns = ["timestamp", "L", "K"]
                dff["timestamp"] = pd.to_datetime(dff["timestamp"], errors="coerce")
                dff["L"] = pd.to_numeric(dff["L"], errors="coerce").clip(0, 1)
                dff["K"] = pd.to_numeric(dff["K"], errors="coerce").clip(0, 1)
                dff = dff.dropna()

                if len(dff) < 5:
                    st.warning("Not enough rows after cleaning.")
                else:
                    fig = go.Figure()

                    fig.add_trace(
                        go.Histogram2d(
                            x=dff["L"], y=dff["K"],
                            nbinsx=35, nbinsy=35,
                            colorscale="Viridis",
                            showscale=True,
                            opacity=0.55,
                            name="Density",
                            hovertemplate="L: %{x:.2f}<br>K: %{y:.2f}<br>count: %{z}<extra></extra>"
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=dff["L"], y=dff["K"],
                            mode="lines+markers",
                            marker=dict(size=5),
                            line=dict(width=2),
                            name="Trajectory",
                            hovertemplate="L: %{x:.2f}<br>K: %{y:.2f}<extra></extra>"
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=[float(dff["L"].iloc[-1])],
                            y=[float(dff["K"].iloc[-1])],
                            mode="markers",
                            marker=dict(size=14, symbol="x", color="black"),
                            name="Now",
                            hoverinfo="skip"
                        )
                    )

                    fig.add_shape(
                        type="rect",
                        x0=zones_params["L_min_vault"], x1=zones_params["L_max_vault"],
                        y0=zones_params["K_min_vault"], y1=zones_params["K_max_vault"],
                        line=dict(width=2, color="#EF4444", dash="dash"),
                        fillcolor="rgba(239,68,68,0.06)",
                        layer="below"
                    )
                    fig.add_shape(
                        type="rect",
                        x0=zones_params["L_min_opt"], x1=zones_params["L_max_opt"],
                        y0=zones_params["K_min_opt"], y1=zones_params["K_max_opt"],
                        line=dict(width=2, color="#FBBF24", dash="dot"),
                        fillcolor="rgba(251,191,36,0.10)",
                        layer="below"
                    )

                    fig.update_layout(
                        title="Operational Control Chart (from your uploaded L/K series)",
                        xaxis_title="L (usage ratio)",
                        yaxis_title="K (capacity ratio)",
                        height=600
                    )
                    fig.update_xaxes(range=[0, 1])
                    fig.update_yaxes(range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Could not parse CSV: {e}")

# -----------------------------
# TAB 3 — Price Shock & Liquidation
# -----------------------------
with tab3:
    st.subheader("Price Shock & Liquidation Feasibility Map")
    st.caption("Carte microstructure: β = taille liquidation / depth, Δ = stress (haircut + basis + price shock).")
    st.plotly_chart(liq_fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Executable depth after haircut ($)", f"${liq_metrics['effective_depth_usd']:,.0f}")
    c2.metric("Liquidation size ($)", f"${liq_metrics['liquidation_size_usd']:,.0f}")
    c3.metric("Δ (stress index)", f"{liq_metrics['delta_current']:.2f}")
    if liq_params["lif_model"].startswith("Proxy"):
        c4.metric("LIF (from LLTV)", f"{liq_metrics['lif_proxy']*100:.1f}%")
    else:
        c4.metric("LIF reference", "5% / 10%")

    st.markdown("---")
    st.write(f"**Verdict:** {liq_verdict}")
    if liq_params["price_shock_pct"] != 0:
        st.info(f"Price shock of {liq_params['price_shock_pct']:+.1f}% applied to basis.")

    with st.expander("How to interpret the map"):
        st.write("- **Safe (vert)**: liquidation rentable.\n- **Marginal (jaune)**: dépend des hypothèses.\n- **Econ-unsafe (rose)**: perte > incentive → bad debt.\n- **Dislocation (bleu)**: collapse de liquidité.")

# -----------------------------
# TAB 4 — Market health monitor
# -----------------------------
with tab4:
    st.subheader("Market Health Monitor")
    left, right = st.columns([1.0, 1.2])

    with left:
        st.plotly_chart(plot_gauge(score), use_container_width=True)
        st.metric("Verdict", verdict)

        st.markdown("### Drivers")
        st.write(f"- **Vault zones health**: `{vault_health:.2f}`\n- **Liquidation health**: `{liq_health:.2f}`\n- **Quality proxy**: `{q_norm:.2f}`")

    with right:
        st.markdown("### What the score means")
        st.write("Score composite (0–100) basé sur vault zones, liquidation feasibility et quality proxy.")
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Optimal", f"{p_opt*100:.1f}%")
        c2.metric("🟡 Acceptable", f"{p_acc*100:.1f}%")
        c3.metric("🔴 Out of Spec", f"{p_out*100:.1f}%")

        st.markdown("---")
        health_row = pd.DataFrame([{
            "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "universe_size": len(uni_z),
            "p_optimal": p_opt,
            "p_acceptable": p_acc,
            "p_out_of_spec": p_out,
            "vault_health_0_1": vault_health,
            "liq_zone_current": liq_zone_current,
            "liq_health_0_1": liq_health,
            "quality_proxy_0_1": q_norm,
            "market_health_score_0_100": score,
            "verdict": verdict,
            "liq_verdict": liq_verdict,
        }])
        st.dataframe(health_row, use_container_width=True, hide_index=True)
        st.download_button("📥 Download health snapshot CSV", data=health_row.to_csv(index=False).encode("utf-8"),
                           file_name="market_health_snapshot.csv", mime="text/csv")

# -----------------------------
# TAB 5 — Oracle Analysis (UI inspirée des captures)
# -----------------------------
with tab5:
    st.subheader("Oracles Analysis: Main Market")
    st.caption("Analyse des divergences entre prix spot et TWAP par token. Données simulées.")

    # Adresse de marché (pour correspondre à la capture)
    st.code("7u3HeHxYDLhnCoErrtycNokbQYbWGzLs6JSDqGAv5PfF", language="text")

    # Sélecteurs
    col1, col2, col3 = st.columns(3)
    with col1:
        token_options = ["SOL", "BTC", "ETH", "USDC", "wstETH", "WBTC", "LDO", "ARB"]
        selected_token = st.selectbox("Select Token", token_options, index=0)
    with col2:
        period_days = st.selectbox("Select Historical Period (Days)", [1, 7, 30], index=0)
    with col3:
        quantile = st.selectbox("Select Quantile", [0.5, 0.75, 0.9, 0.95, 0.99], index=4)

    # Génération des données
    df_oracle = generate_oracle_data(selected_token, period_days)

    # Graphique de la divergence
    fig = px.line(df_oracle, x="ts", y="twap_divergence_period",
                  title=f"TWAP/Spot Divergence - {selected_token}",
                  labels={"ts": "Time", "twap_divergence_period": "Divergence (%)"})
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Affichage du quantile
    q_value = df_oracle["twap_divergence_period"].quantile(quantile)
    st.metric(f"Quantile {quantile:.2f}", f"{q_value:.4f}%")

    # Tableau des 15 dernières entrées (ordre décroissant)
    st.dataframe(
        df_oracle.tail(15).sort_values("ts", ascending=False).reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

    # Histogramme optionnel
    with st.expander("Show distribution histogram"):
        fig_hist = px.histogram(df_oracle, x="twap_divergence_period", nbins=50,
                                 title=f"Distribution des divergences - {selected_token}")
        lower_q = df_oracle["twap_divergence_period"].quantile(0.01)
        upper_q = df_oracle["twap_divergence_period"].quantile(0.99)
        fig_hist.add_vline(x=lower_q, line_dash="dash", line_color="red",
                           annotation_text=f"Q1% = {lower_q:.2f}%")
        fig_hist.add_vline(x=upper_q, line_dash="dash", line_color="green",
                           annotation_text=f"Q99% = {upper_q:.2f}%")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.caption("⚠️ Données simulées pour illustration. En production, utiliser des APIs de prix on-chain.")

# -----------------------------
# TAB 6 — Loans Analysis (simulé)
# -----------------------------
with tab6:
    st.subheader("Loans Analysis – Main Market (simulated data)")
    st.caption("Simplified visual: each point is a loan, sized by net value, colored by health status.")

    loans_df = generate_mock_loans(50)

    colA, colB, colC = st.columns(3)
    with colA:
        min_deposit = st.number_input("Min deposit value (USD)", 0.0, value=0.01, step=0.01)
    with colB:
        min_borrow = st.number_input("Min borrow value (USD)", 0.0, value=0.0, step=0.01)
    with colC:
        top_n = st.slider("Top N loans", 10, 100, 30)

    sort_by = st.selectbox("Sort by", ["dist_to_liqu", "total_debt", "net_value", "unhealthy"], index=0)

    filtered = loans_df[
        (loans_df["total_borrow"] >= min_borrow) &
        (loans_df["total_debt"] >= min_deposit)
    ].copy()

    filtered = filtered.sort_values(sort_by, ascending=False).head(top_n)

    fig = px.scatter(
        filtered,
        x="total_debt",
        y="dist_to_liqu",
        size="net_value",
        color="unhealthy",
        color_continuous_scale=["green", "red"],
        hover_data={
            "owner": True,
            "pubkey": True,
            "current_ltv": ":.2f",
            "max_ltv": ":.2f",
            "total_borrow": ":.2f",
            "net_value": ":.2f"
        },
        labels={"total_debt": "Total Debt (USD)", "dist_to_liqu": "Distance to Liquidation", "unhealthy": "Unhealthy"},
        title="Loan Health Scatter"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View filtered loan data"):
        display_cols = ["owner", "pubkey", "current_ltv", "max_ltv", "unhealthy",
                        "dist_to_liqu", "total_debt", "total_borrow", "net_value", "url"]
        st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

st.caption("Notes: capacity_ratio uses TVL vs historical max TVL when available (proxy). usage_ratio is protocol-level best-effort. Oracle tab uses simulated data for illustration.")
