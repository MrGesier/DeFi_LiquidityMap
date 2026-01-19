# Liquidation Feasibility in DeFi  
**Interactive risk visualization for LLTV, LIF & stressed liquidation execution**

---

## Overview

This Streamlit app visualizes **liquidation feasibility in DeFi lending markets** under realistic on-chain stress conditions.

It is designed to answer one core risk question:

> **Can forced liquidation remain economically executable under stress, given LLTV-implied incentives?**

The app explicitly models:
- Convex slippage from market microstructure
- Liquidity haircuts under stress
- Oracle ↔ spot basis erosion
- The relationship between **LLTV → Liquidation Incentive Factor (LIF)**
- Structural conditions leading to **non-liquidation and bad debt**

This framework aligns with **Steakhouse / Morpho-style risk underwriting**, where execution feasibility dominates price forecasting.

---

## Key Concepts Visualized

### Axes
- **X-axis — Liquidation Size / Executable Depth (%)**  
  Size of the forced liquidation relative to *real, sellable* market depth.

- **Y-axis — Stress Index Δ**  
  Composite stress variable including:
  - Liquidity haircut (LP withdrawal)
  - Oracle ↔ spot basis
  - Congestion / MEV effects

---

### Zones
- **Executable & Profitable**  
  Liquidation incentives exceed effective execution losses → liquidators act.

- **Marginal Zone**  
  Liquidation outcome is sensitive to incentives and assumptions.

- **LLTV too high → LIF too low**  
  Effective losses exceed incentives → liquidators abstain → bad debt risk.

- **Dislocation Zone**  
  Liquidity collapses (pool drains, execution failure).

---

## Parameters & Sliders

The app lets you interactively adjust:

### Market & Execution
- Liquidation size (% of pool / depth)
- Executable market depth ($)
- Convexity of slippage
- Liquidity haircut severity
- Oracle ↔ spot basis penalty

### Risk Configuration
- **LLTV** (Liquidation Loan-To-Value)
- Derived **Liquidation Incentive Factor (LIF)**
- Stress amplification parameters

All changes update in real time:
- Slippage contours
- LIF thresholds (highlighted in red)
- Market dislocation boundary
- Current scenario position

---

## Why This Matters (Risk Perspective)

Traditional risk metrics implicitly assume:
- Continuous liquidity
- Linear execution costs
- Guaranteed liquidation

This app makes explicit that:
- Slippage is **convex**
- Stress effects are **non-linear**
- Liquidation is **optional**, not guaranteed

> **If effective liquidation loss > LIF, liquidation does not occur.**  
> → Losses are socialized to lenders.

This is a **structural risk**, not a tail event.

---

## Installation

### Requirements
- Python 3.9+
- Streamlit
- NumPy
- Plotly

### Install & Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
