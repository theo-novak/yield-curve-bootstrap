"""
Yield Curve Dashboard — Streamlit app.

Launch:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src package importable when running from the project root
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[1] / ".env")

from yieldcurve.curve.bootstrap import BootstrappedCurve
from yieldcurve.curve.interpolation import InterpolatedCurve, InterpolationMethod
from yieldcurve.curve.nelson_siegel import NelsonSiegelCurve, NelsonSiegelSvenssonCurve
from yieldcurve.data.fetcher import fetch_treasury_yields, yields_for_date
from yieldcurve.data.storage import cached_date_range, load_yields, save_yields
from yieldcurve.risk.instruments import FixedRateBond
from yieldcurve.risk.scenarios import ShockEngine, ShockType
from yieldcurve.viz.plots import (
    plot_zero_curve,
    plot_discount_factors,
    plot_forward_curve,
    plot_interpolation_comparison,
    plot_scenario_curves,
    plot_dv01_ladder,
    plot_yield_history,
    plot_3d_surface,
)

# ─── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Yield Curve Explorer",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Yield Curve Explorer")
st.caption("Bootstrap zero curves from US Treasury CMT data · Interpolation comparison · Shock scenarios · Bond analytics")

# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Data")

    api_key = st.text_input(
        "FRED API Key",
        value=os.environ.get("FRED_API_KEY", ""),
        type="password",
        help="Free key at https://fred.stlouisfed.org/docs/api/fred/",
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("History start", value=date(2000, 1, 1))
    with col2:
        fetch_end = st.date_input("History end", value=date.today())

    if st.button("Fetch & cache data", use_container_width=True):
        if not api_key:
            st.error("Enter your FRED API key first.")
        else:
            with st.spinner("Fetching from FRED…"):
                try:
                    df_raw = fetch_treasury_yields(start_date, fetch_end, api_key=api_key)
                    save_yields(df_raw)
                    st.success(f"Cached {len(df_raw):,} rows ({df_raw.index[0].date()} – {df_raw.index[-1].date()})")
                except Exception as e:
                    st.error(f"Fetch failed: {e}")

    st.divider()
    st.header("Curve date")

    min_dt, max_dt = cached_date_range()
    if min_dt is None:
        st.info("No cached data. Fetch data above first.")
        selected_date = date.today() - timedelta(days=1)
    else:
        selected_date = st.date_input(
            "As-of date",
            value=max_dt.date(),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
        )

    st.divider()
    st.header("Interpolation")
    interp_choice = st.selectbox(
        "Method",
        options=[m.value for m in InterpolationMethod],
        index=3,
        format_func=lambda v: {
            "linear_zero": "Linear on zero rates",
            "cubic_zero": "Cubic spline on zero rates",
            "log_linear_df": "Log-linear on discount factors",
            "cubic_log_df": "Cubic spline on log-DF",
        }[v],
    )

    st.divider()
    st.header("Shock scenarios")
    shock_magnitude = st.slider("Magnitude (bp)", 10, 300, 100, step=10)
    selected_shocks = st.multiselect(
        "Scenarios",
        options=[s.value for s in ShockType],
        default=["parallel_up", "parallel_down", "bear_steepener", "bull_flattener"],
        format_func=lambda v: v.replace("_", " ").title(),
    )

    st.divider()
    st.header("Bond analytics")
    bond_coupon = st.number_input("Coupon rate (%)", 0.0, 20.0, 4.0, 0.25) / 100
    bond_maturity = st.number_input("Maturity (years)", 0.5, 30.0, 10.0, 0.5)


# ─── Load curve data ─────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_curve_data(as_of: str) -> pd.Series | None:
    df = load_yields()
    if df.empty:
        return None
    try:
        return yields_for_date(df, as_of)
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_history() -> pd.DataFrame:
    return load_yields()


par_yields_series = load_curve_data(str(selected_date))

if par_yields_series is None or par_yields_series.empty:
    st.warning(
        "No data loaded. Use the sidebar to fetch Treasury data from FRED first, "
        "then pick a curve date."
    )
    st.stop()

# ─── Bootstrap ───────────────────────────────────────────────────────────────

curve_ts = pd.Timestamp(selected_date)
bootstrap = BootstrappedCurve.from_series(par_yields_series, curve_date=curve_ts)
interp_curve = InterpolatedCurve(bootstrap, InterpolationMethod(interp_choice))

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab_curve, tab_interp, tab_scenarios, tab_bond, tab_history, tab_surface = st.tabs([
    "📐 Zero Curve",
    "🔀 Interpolation",
    "⚡ Shock Scenarios",
    "🏦 Bond Analytics",
    "📅 History",
    "🌐 3-D Surface",
])

# ═══ Tab 1: Zero Curve ═══════════════════════════════════════════════════════

with tab_curve:
    st.subheader(f"Zero-Coupon Curve — {selected_date}")

    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.plotly_chart(
            plot_zero_curve([interp_curve], labels=[f"{selected_date} ({interp_choice})"]),
            use_container_width=True,
        )
        st.plotly_chart(plot_forward_curve(interp_curve), use_container_width=True)

    with col_right:
        st.subheader("Bootstrap table")
        df_table = bootstrap.to_dataframe().copy()
        df_table["zero_rate"] = (df_table["zero_rate"] * 100).round(4)
        df_table["discount_factor"] = df_table["discount_factor"].round(6)
        df_table["continuous_rate"] = (df_table["continuous_rate"] * 100).round(4)
        df_table.columns = ["Tenor (yr)", "Zero Rate (%)", "DF", "Cont. Rate (%)"]
        st.dataframe(df_table, use_container_width=True, hide_index=True)

        st.subheader("Discount factors")
        st.plotly_chart(plot_discount_factors([interp_curve]), use_container_width=True)

    # Nelson-Siegel fit
    with st.expander("Nelson-Siegel parametric fit"):
        ns_curve = NelsonSiegelCurve.fit(bootstrap.tenors, bootstrap.zero_rates)
        t_grid = np.linspace(bootstrap.tenors[0], bootstrap.tenors[-1], 300)
        ns_z = ns_curve.zero_rate(t_grid) * 100
        boot_z = interp_curve.zero_rate(t_grid) * 100

        import plotly.graph_objects as go
        fig_ns = go.Figure()
        fig_ns.add_trace(go.Scatter(x=t_grid, y=boot_z, mode="lines", name="Bootstrap (interpolated)"))
        fig_ns.add_trace(go.Scatter(x=t_grid, y=ns_z, mode="lines", name="Nelson-Siegel fit", line=dict(dash="dash")))
        fig_ns.update_layout(template="plotly_white", xaxis_title="Maturity (years)", yaxis_title="Zero Rate (%)", hovermode="x unified")
        st.plotly_chart(fig_ns, use_container_width=True)

        params = ns_curve.parameter_table()
        st.table(pd.DataFrame.from_dict(params, orient="index", columns=["Value"]).round(6))


# ═══ Tab 2: Interpolation Comparison ════════════════════════════════════════

with tab_interp:
    st.subheader("Interpolation Method Comparison")
    st.markdown(
        "All four methods fit the same bootstrap nodes (markers). "
        "Differences are most visible in the short end and between data-sparse tenors."
    )
    st.plotly_chart(plot_interpolation_comparison(bootstrap), use_container_width=True)

    st.subheader("Forward curve by interpolation method")
    from yieldcurve.curve.interpolation import InterpolatedCurve, InterpolationMethod
    import plotly.graph_objects as go

    fig_fwd = go.Figure()
    method_labels = {
        InterpolationMethod.LINEAR_ZERO: "Linear (zero)",
        InterpolationMethod.CUBIC_ZERO: "Cubic (zero)",
        InterpolationMethod.LOG_LINEAR_DF: "Log-linear (DF)",
        InterpolationMethod.CUBIC_LOG_DF: "Cubic log-DF",
    }
    import plotly.express as px
    palette = px.colors.qualitative.Set2
    t_nodes = bootstrap.tenors
    t_grid = np.linspace(float(t_nodes[0]), float(t_nodes[-1]), 200)
    step = t_grid[1] - t_grid[0]

    for i, (method, label) in enumerate(method_labels.items()):
        c = InterpolatedCurve(bootstrap, method)
        fwd = np.array([c.forward_rate(max(t - step, 1e-4), t) for t in t_grid]) * 100
        fig_fwd.add_trace(go.Scatter(x=t_grid, y=fwd, mode="lines", name=label,
                                     line=dict(color=palette[i], width=1.5)))

    fig_fwd.update_layout(template="plotly_white", title="Implied Forward Rates by Method",
                           xaxis_title="Maturity (years)", yaxis_title="Forward Rate (%)", hovermode="x unified")
    st.plotly_chart(fig_fwd, use_container_width=True)


# ═══ Tab 3: Shock Scenarios ══════════════════════════════════════════════════

with tab_scenarios:
    st.subheader(f"Shock Scenarios — ±{shock_magnitude} bp")

    shocked = {
        name: ShockEngine.apply(interp_curve, ShockType(name), shock_magnitude)
        for name in selected_shocks
    }

    st.plotly_chart(plot_scenario_curves(interp_curve, shocked), use_container_width=True)

    if shocked:
        st.subheader("Scenario zero rates at key tenors")
        key_tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
        key_labels = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

        rows = {"Base": [interp_curve.zero_rate(t) * 100 for t in key_tenors]}
        for name, c in shocked.items():
            rows[name.replace("_", " ").title()] = [c.zero_rate(t) * 100 for t in key_tenors]

        df_scenarios = pd.DataFrame(rows, index=key_labels).T.round(4)
        st.dataframe(df_scenarios, use_container_width=True)


# ═══ Tab 4: Bond Analytics ═══════════════════════════════════════════════════

with tab_bond:
    st.subheader(f"Fixed-Rate Bond — {bond_coupon*100:.2f}% coupon, {bond_maturity}Y maturity")

    bond = FixedRateBond(coupon_rate=bond_coupon, maturity=bond_maturity)
    report = bond.risk_report(interp_curve)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${report['price']:.4f}")
    c2.metric("YTM", f"{report['ytm_pct']:.4f}%")
    c3.metric("Modified Duration", f"{report['modified_duration']:.3f} yrs")
    c4.metric("DV01 (/$100 par)", f"${report['dv01']:.4f}")

    c5, c6 = st.columns(2)
    c5.metric("Macaulay Duration", f"{report['macaulay_duration']:.3f} yrs")
    c6.metric("Convexity", f"{report['convexity']:.3f}")

    st.subheader("P&L estimates under rate shocks")
    shock_sizes = [-200, -100, -50, -25, +25, +50, +100, +200]
    pnl_rows = []
    for bps in shock_sizes:
        est = bond.price_change_estimate(interp_curve, bps)
        pnl_rows.append({
            "Shock (bp)": bps,
            "Duration P&L ($)": round(est["duration_pnl"], 4),
            "Convexity P&L ($)": round(est["convexity_pnl"], 6),
            "Total P&L ($)": round(est["total_pnl"], 4),
            "% Change": round(est["pct_change"], 4),
        })
    st.dataframe(pd.DataFrame(pnl_rows), use_container_width=True, hide_index=True)

    st.subheader("Key-rate DV01 ladder")
    dv01_ladder = {}
    for lbl in ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]:
        shifted = ShockEngine.key_rate_shift(interp_curve, lbl, shift_bps=1.0)
        dv01_ladder[lbl] = bond.price(shifted) - report["price"]

    st.plotly_chart(plot_dv01_ladder(dv01_ladder), use_container_width=True)


# ═══ Tab 5: History ══════════════════════════════════════════════════════════

with tab_history:
    st.subheader("Treasury Yield History")
    df_hist = load_history()
    if df_hist.empty:
        st.info("Fetch historical data first using the sidebar.")
    else:
        tenor_select = st.multiselect(
            "Tenors to display",
            options=list(df_hist.columns),
            default=["2Y", "5Y", "10Y", "30Y"],
        )
        if tenor_select:
            st.plotly_chart(
                plot_yield_history(df_hist, tenor_select),
                use_container_width=True,
            )

        st.subheader("Spread: 10Y – 2Y (recession indicator)")
        if "10Y" in df_hist.columns and "2Y" in df_hist.columns:
            spread = (df_hist["10Y"] - df_hist["2Y"]).dropna()
            import plotly.graph_objects as go
            fig_spread = go.Figure()
            fig_spread.add_trace(go.Scatter(
                x=spread.index, y=spread.values, mode="lines", name="10Y-2Y spread",
                fill="tozeroy",
                line=dict(color="steelblue"),
            ))
            fig_spread.add_hline(y=0, line_color="red", line_dash="dash")
            fig_spread.update_layout(template="plotly_white", xaxis_title="Date",
                                     yaxis_title="Spread (%)", hovermode="x unified")
            st.plotly_chart(fig_spread, use_container_width=True)


# ═══ Tab 6: 3-D Surface ══════════════════════════════════════════════════════

with tab_surface:
    st.subheader("Yield Curve Surface")
    df_hist = load_history()
    if df_hist.empty:
        st.info("Fetch historical data first.")
    else:
        max_rows = st.slider("Number of dates (most recent)", 50, min(2000, len(df_hist)), 500, step=50)
        df_slice = df_hist.tail(max_rows)
        st.plotly_chart(plot_3d_surface(df_slice), use_container_width=True)
