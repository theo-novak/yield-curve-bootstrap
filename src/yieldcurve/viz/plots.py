"""Reusable Plotly figures for yield curve analysis."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

_PALETTE = px.colors.qualitative.Set2
_TEMPLATE = "plotly_white"


def _curve_label(curve) -> str:
    d = getattr(curve, "curve_date", None)
    return d.strftime("%Y-%m-%d") if d is not None else "Curve"


def plot_zero_curve(
    curves: list,
    labels: list[str] | None = None,
    n_points: int = 300,
    title: str = "Zero-Coupon Spot Rates",
) -> go.Figure:
    fig = go.Figure()
    for i, curve in enumerate(curves):
        lbl = (labels[i] if labels else _curve_label(curve))
        t_max = float(curve.base.tenors[-1]) if hasattr(curve, "base") else float(curve.tenors[-1])
        t_nodes = curve.base.tenors if hasattr(curve, "base") else curve.tenors
        t_min = float(t_nodes[0])
        t_grid = np.linspace(t_min, t_max, n_points)

        z = np.asarray(curve.zero_rate(t_grid), dtype=float) * 100

        fig.add_trace(go.Scatter(
            x=t_grid, y=z, mode="lines", name=lbl,
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
        ))
        # Overlay bootstrap nodes if available
        if hasattr(curve, "base"):
            fig.add_trace(go.Scatter(
                x=t_nodes, y=curve.base.zero_rates * 100,
                mode="markers", name=f"{lbl} (nodes)",
                marker=dict(symbol="circle-open", size=8, color=_PALETTE[i % len(_PALETTE)]),
                showlegend=False,
            ))

    fig.update_layout(
        template=_TEMPLATE, title=title,
        xaxis_title="Maturity (years)", yaxis_title="Zero Rate (%)",
        hovermode="x unified",
    )
    return fig


def plot_discount_factors(
    curves: list,
    labels: list[str] | None = None,
    n_points: int = 300,
) -> go.Figure:
    fig = go.Figure()
    for i, curve in enumerate(curves):
        lbl = labels[i] if labels else _curve_label(curve)
        t_nodes = curve.base.tenors if hasattr(curve, "base") else curve.tenors
        t_max = float(t_nodes[-1])
        t_min = float(t_nodes[0])
        t_grid = np.linspace(t_min, t_max, n_points)
        df = np.asarray(curve.discount_factor(t_grid), dtype=float)

        fig.add_trace(go.Scatter(
            x=t_grid, y=df, mode="lines", name=lbl,
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
        ))

    fig.update_layout(
        template=_TEMPLATE, title="Discount Factors",
        xaxis_title="Maturity (years)", yaxis_title="Discount Factor",
        hovermode="x unified",
    )
    return fig


def plot_forward_curve(curve, n_points: int = 200) -> go.Figure:
    t_nodes = curve.base.tenors if hasattr(curve, "base") else curve.tenors
    t_max = float(t_nodes[-1])
    step = t_max / n_points
    t_grid = np.arange(step, t_max, step)

    fwd = np.array([curve.forward_rate(max(t - step, 1e-4), t) for t in t_grid]) * 100
    zero = np.asarray(curve.zero_rate(t_grid), dtype=float) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_grid, y=zero, mode="lines", name="Zero rate",
                             line=dict(color=_PALETTE[0], width=2)))
    fig.add_trace(go.Scatter(x=t_grid, y=fwd, mode="lines", name="Instantaneous fwd",
                             line=dict(color=_PALETTE[1], width=1.5, dash="dash")))

    fig.update_layout(
        template=_TEMPLATE, title="Zero vs Forward Rates",
        xaxis_title="Maturity (years)", yaxis_title="Rate (%)",
        hovermode="x unified",
    )
    return fig


def plot_interpolation_comparison(
    base_bootstrap,
    n_points: int = 300,
) -> go.Figure:
    from ..curve.interpolation import InterpolatedCurve, InterpolationMethod

    methods = list(InterpolationMethod)
    labels = ["Linear (zero)", "Cubic spline (zero)", "Log-linear (DF)", "Cubic spline (log-DF)"]
    t_nodes = base_bootstrap.tenors
    t_grid = np.linspace(float(t_nodes[0]), float(t_nodes[-1]), n_points)

    fig = go.Figure()
    for i, (method, label) in enumerate(zip(methods, labels)):
        curve = InterpolatedCurve(base_bootstrap, method)
        z = np.asarray(curve.zero_rate(t_grid), dtype=float) * 100
        fig.add_trace(go.Scatter(
            x=t_grid, y=z, mode="lines", name=label,
            line=dict(color=_PALETTE[i], width=2),
        ))

    fig.add_trace(go.Scatter(
        x=t_nodes, y=base_bootstrap.zero_rates * 100,
        mode="markers", name="Bootstrap nodes",
        marker=dict(symbol="x", size=10, color="black"),
    ))
    fig.update_layout(
        template=_TEMPLATE, title="Interpolation Method Comparison",
        xaxis_title="Maturity (years)", yaxis_title="Zero Rate (%)",
        hovermode="x unified",
    )
    return fig


def plot_scenario_curves(
    base_curve,
    shocked_curves: dict[str, object],
    n_points: int = 300,
) -> go.Figure:
    t_nodes = base_curve.base.tenors if hasattr(base_curve, "base") else base_curve.tenors
    t_grid = np.linspace(float(t_nodes[0]), float(t_nodes[-1]), n_points)

    fig = go.Figure()

    # Base curve
    z_base = np.asarray(base_curve.zero_rate(t_grid), dtype=float) * 100
    fig.add_trace(go.Scatter(
        x=t_grid, y=z_base, mode="lines", name="Base",
        line=dict(color="black", width=3),
    ))

    for i, (name, curve) in enumerate(shocked_curves.items()):
        z = np.asarray(curve.zero_rate(t_grid), dtype=float) * 100
        fig.add_trace(go.Scatter(
            x=t_grid, y=z, mode="lines", name=name.replace("_", " ").title(),
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=1.5, dash="dot"),
        ))

    fig.update_layout(
        template=_TEMPLATE, title="Rate Shock Scenarios",
        xaxis_title="Maturity (years)", yaxis_title="Zero Rate (%)",
        hovermode="x unified",
    )
    return fig


def plot_dv01_ladder(dv01_by_tenor: dict[str, float]) -> go.Figure:
    labels = list(dv01_by_tenor.keys())
    values = list(dv01_by_tenor.values())
    colors = [_PALETTE[1] if v >= 0 else _PALETTE[2] for v in values]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"${v:.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        template=_TEMPLATE, title="Key-Rate DV01 Ladder (per $100 par)",
        xaxis_title="Tenor", yaxis_title="DV01 ($)",
    )
    return fig


def plot_yield_history(
    df,
    tenors: list[str] | None = None,
    title: str = "Treasury Yield History",
) -> go.Figure:
    cols = tenors or list(df.columns)
    fig = go.Figure()
    for i, col in enumerate(cols):
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], mode="lines", name=col,
            line=dict(color=_PALETTE[i % len(_PALETTE)]),
        ))
    fig.update_layout(
        template=_TEMPLATE, title=title,
        xaxis_title="Date", yaxis_title="Yield (%)",
        hovermode="x unified",
    )
    return fig


def plot_3d_surface(df, title: str = "Yield Surface") -> go.Figure:
    """3-D surface of Treasury yields over time."""
    maturity_order = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    cols = [c for c in maturity_order if c in df.columns]
    tenor_map = {
        "1M": 1/12, "3M": 0.25, "6M": 0.5,
        "1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5,
        "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30,
    }
    z_mat = df[cols].values
    x = [tenor_map[c] for c in cols]
    y = df.index

    fig = go.Figure(go.Surface(
        z=z_mat, x=x, y=list(range(len(y))),
        colorscale="RdYlGn_r",
        colorbar=dict(title="Yield (%)"),
    ))
    # Sparse y-axis tick labels
    step = max(1, len(y) // 10)
    fig.update_layout(
        template=_TEMPLATE, title=title,
        scene=dict(
            xaxis_title="Maturity (years)",
            yaxis_title="Date",
            zaxis_title="Yield (%)",
            yaxis=dict(
                tickvals=list(range(0, len(y), step)),
                ticktext=[str(y[i].date()) for i in range(0, len(y), step)],
            ),
        ),
    )
    return fig
