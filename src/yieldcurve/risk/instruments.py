"""
Fixed-income instrument pricing and risk analytics from a discount curve.

Conventions:
  - All bonds: semiannual coupons, par = 100
  - t=0 is the curve date; settlement not adjusted
  - DV01 = price change per +1 bp shift in yield
  - Modified duration and convexity use yield-to-maturity basis
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FixedRateBond:
    """
    A generic fixed-rate bond with semiannual coupon payments.

    Parameters
    ----------
    coupon_rate : float   Annual coupon rate (decimal, e.g. 0.04 for 4%)
    maturity    : float   Years to maturity (e.g. 10.0)
    par         : float   Par value (default 100)
    """

    coupon_rate: float
    maturity: float
    par: float = 100.0

    def _cash_flows(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (times_in_years, cash_flows)."""
        periods = round(self.maturity * 2)
        times = np.arange(1, periods + 1) * 0.5
        coupon = self.coupon_rate / 2 * self.par
        flows = np.full(periods, coupon)
        flows[-1] += self.par
        return times, flows

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(self, curve) -> float:
        """Price the bond by discounting cash flows from *curve*."""
        times, flows = self._cash_flows()
        dfs = np.array([float(curve.discount_factor(t)) for t in times])
        return float(np.sum(flows * dfs))

    def yield_to_maturity(self, curve) -> float:
        """
        Solve for the flat yield y (semiannual compounding) that reprices the bond.
        """
        from scipy.optimize import brentq

        px = self.price(curve)
        times, flows = self._cash_flows()

        def pv_diff(y):
            dfs = (1.0 + y / 2.0) ** (-times * 2)
            return float(np.sum(flows * dfs)) - px

        try:
            ytm = brentq(pv_diff, -0.5, 5.0, xtol=1e-10)
        except ValueError:
            ytm = float("nan")
        return ytm

    # ------------------------------------------------------------------
    # Duration & convexity
    # ------------------------------------------------------------------

    def macaulay_duration(self, curve) -> float:
        """Macaulay duration in years (semiannual coupon convention)."""
        times, flows = self._cash_flows()
        dfs = np.array([float(curve.discount_factor(t)) for t in times])
        pv_flows = flows * dfs
        price = float(np.sum(pv_flows))
        return float(np.sum(times * pv_flows) / price)

    def modified_duration(self, curve) -> float:
        """Modified duration in years."""
        y = self.yield_to_maturity(curve)
        mac = self.macaulay_duration(curve)
        return mac / (1.0 + y / 2.0)

    def dv01(self, curve) -> float:
        """Dollar Value of 1 basis point (per $100 par)."""
        return self.modified_duration(curve) * self.price(curve) * 0.0001

    def convexity(self, curve) -> float:
        """
        Convexity (semiannual coupon convention) — in years².
        ΔP ≈ -MD·P·Δy + ½·Convexity·P·(Δy)²
        """
        y = self.yield_to_maturity(curve)
        y2 = y / 2.0
        times, flows = self._cash_flows()
        periods = times * 2  # number of half-periods
        px = self.price(curve)

        conv = sum(
            cf * t_p * (t_p + 1) / (1.0 + y2) ** (t_p + 2)
            for cf, t_p in zip(flows, periods)
        )
        return float(conv / px / 4.0)  # annualised

    # ------------------------------------------------------------------
    # Full risk report
    # ------------------------------------------------------------------

    def risk_report(self, curve) -> dict[str, float]:
        px = self.price(curve)
        ytm = self.yield_to_maturity(curve)
        mac = self.macaulay_duration(curve)
        md = mac / (1.0 + ytm / 2.0)
        dv01 = md * px * 0.0001
        conv = self.convexity(curve)

        return {
            "price": px,
            "ytm_pct": ytm * 100,
            "macaulay_duration": mac,
            "modified_duration": md,
            "dv01": dv01,
            "convexity": conv,
            "coupon_rate_pct": self.coupon_rate * 100,
            "maturity_years": self.maturity,
        }

    def price_change_estimate(
        self, curve, rate_shock_bps: float
    ) -> dict[str, float]:
        """Approximate price change using modified duration + convexity."""
        dy = rate_shock_bps / 10_000
        px = self.price(curve)
        md = self.modified_duration(curve)
        conv = self.convexity(curve)

        delta_dur = -md * px * dy
        delta_conv = 0.5 * conv * px * dy ** 2
        total = delta_dur + delta_conv

        return {
            "duration_pnl": delta_dur,
            "convexity_pnl": delta_conv,
            "total_pnl": total,
            "pct_change": total / px * 100,
        }


def bond_portfolio_dv01_ladder(
    bonds: list[FixedRateBond], curve
) -> dict[str, float]:
    """
    Approximate DV01 at each key tenor bucket by bumping one node at a time (+1 bp).
    Returns {tenor_label: dv01_contribution}.
    """
    from .scenarios import ShockEngine, ShockType

    base_prices = [b.price(curve) for b in bonds]
    buckets = {}

    for tenor_label in ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]:
        shocked = ShockEngine.key_rate_shift(curve, tenor_label, shift_bps=1.0)
        dv01_sum = sum(
            b.price(shocked) - bp for b, bp in zip(bonds, base_prices)
        )
        buckets[tenor_label] = dv01_sum

    return buckets
