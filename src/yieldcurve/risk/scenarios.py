"""
Rate shock scenarios applied to a bootstrapped or interpolated curve.

Scenario types
--------------
PARALLEL_UP / PARALLEL_DOWN     All tenors shift uniformly.
BEAR_STEEPENER                  Short end rises more; long end rises less.
BULL_STEEPENER                  Short end falls more; long end falls less.
BEAR_FLATTENER                  Short end rises less; long end rises more.
BULL_FLATTENER                  Short end falls less; long end falls more.
TWIST                           2Y rises, 10Y falls, 5Y unchanged (butterfly).
KEY_RATE                        Single-tenor perturbation; all others unchanged.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from ..curve.bootstrap import BootstrappedCurve


class ShockType(str, Enum):
    PARALLEL_UP = "parallel_up"
    PARALLEL_DOWN = "parallel_down"
    BEAR_STEEPENER = "bear_steepener"
    BULL_STEEPENER = "bull_steepener"
    BEAR_FLATTENER = "bear_flattener"
    BULL_FLATTENER = "bull_flattener"
    TWIST = "twist"


# Shock profiles: (short_bps, long_bps) — applied continuously across the curve
# using a smooth blending function.  Short anchor = 2Y, Long anchor = 10Y.
_SHOCK_PROFILES: dict[ShockType, tuple[float, float]] = {
    ShockType.PARALLEL_UP:     (100, 100),
    ShockType.PARALLEL_DOWN:   (-100, -100),
    ShockType.BEAR_STEEPENER:  (150, 50),
    ShockType.BULL_STEEPENER:  (-150, -50),
    ShockType.BEAR_FLATTENER:  (50, 150),
    ShockType.BULL_FLATTENER:  (-50, -150),
    ShockType.TWIST:           (75, -75),   # 2Y +75bp, 10Y -75bp, 5Y≈0
}


class ShockedCurve:
    """
    A thin wrapper that applies an additive spread (in bp terms) to a base curve's
    zero rates when computing discount factors.
    """

    def __init__(self, base, shift_at_tenor: dict[float, float]) -> None:
        """
        base: BootstrappedCurve or InterpolatedCurve
        shift_at_tenor: {tenor_years: shift_in_decimal}  (+ = up in rate)
        """
        self._base = base
        _t = np.array(sorted(shift_at_tenor.keys()))
        _s = np.array([shift_at_tenor[k] for k in _t])
        self._shift_tenors = _t
        self._shift_values = _s
        self.curve_date = getattr(base, "curve_date", None)

    def _shift(self, t: np.ndarray) -> np.ndarray:
        """Interpolate the shock profile at arbitrary tenors."""
        return np.interp(t, self._shift_tenors, self._shift_values, left=self._shift_values[0], right=self._shift_values[-1])

    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        scalar = np.isscalar(t)
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        base_z = np.atleast_1d(np.asarray(self._base.zero_rate(t_arr), dtype=float))
        result = base_z + self._shift(t_arr)
        return float(result[0]) if scalar else result

    def discount_factor(self, t: float | np.ndarray) -> float | np.ndarray:
        z = self.zero_rate(t)
        t_arr = np.asarray(t, dtype=float)
        return (1.0 + z) ** (-t_arr)

    def continuous_zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        df = self.discount_factor(t)
        return -np.log(df) / np.asarray(t, dtype=float)

    def forward_rate(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            raise ValueError("t2 > t1")
        return -np.log(self.discount_factor(t2) / self.discount_factor(t1)) / (t2 - t1)


class ShockEngine:
    """Factory for creating shocked versions of a curve."""

    @staticmethod
    def apply(
        base,
        shock_type: ShockType,
        magnitude_bps: float = 100.0,
    ) -> ShockedCurve:
        """
        Apply a named scenario scaled by *magnitude_bps*.
        The default profile is ±100 bp for parallel; other scenarios scale proportionally.
        """
        short_bps, long_bps = _SHOCK_PROFILES[shock_type]
        scale = magnitude_bps / 100.0

        short_shift = short_bps * scale / 10_000
        long_shift = long_bps * scale / 10_000

        t_nodes = np.array([0.25, 2.0, 10.0, 30.0])
        s_nodes = np.array([
            short_shift,   # 3M anchored to short
            short_shift,   # 2Y = short
            long_shift,    # 10Y = long
            long_shift,    # 30Y anchored to long
        ])
        shift_map = dict(zip(t_nodes, s_nodes))
        return ShockedCurve(base, shift_map)

    @staticmethod
    def parallel(base, shift_bps: float) -> ShockedCurve:
        """Uniform shift across all tenors."""
        shift = shift_bps / 10_000
        t_nodes = np.array([0.0, 30.0])
        shift_map = {0.0: shift, 30.0: shift}
        return ShockedCurve(base, shift_map)

    @staticmethod
    def key_rate_shift(base, tenor_label: str, shift_bps: float = 1.0) -> ShockedCurve:
        """
        Bump a single key-rate tenor by *shift_bps*; taper linearly to adjacent nodes.
        Standard tenors: '1Y','2Y','3Y','5Y','7Y','10Y','20Y','30Y'.
        """
        label_to_years = {
            "1M": 1/12, "3M": 0.25, "6M": 0.5,
            "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "5Y": 5.0,
            "7Y": 7.0, "10Y": 10.0, "20Y": 20.0, "30Y": 30.0,
        }
        ALL_TENORS = sorted(label_to_years.values())

        target = label_to_years[tenor_label]
        idx = ALL_TENORS.index(target)
        lo = ALL_TENORS[idx - 1] if idx > 0 else 0.0
        hi = ALL_TENORS[idx + 1] if idx < len(ALL_TENORS) - 1 else ALL_TENORS[-1] + 5

        shift = shift_bps / 10_000
        shift_map = {lo: 0.0, target: shift, hi: 0.0}
        # zero shift outside the window
        if lo > 0.0:
            shift_map[0.0] = 0.0
        shift_map[ALL_TENORS[-1]] = 0.0

        return ShockedCurve(base, shift_map)

    @staticmethod
    def all_scenarios(base, magnitude_bps: float = 100.0) -> dict[str, ShockedCurve]:
        """Return a dict of all named scenario curves."""
        return {
            shock_type.value: ShockEngine.apply(base, shock_type, magnitude_bps)
            for shock_type in ShockType
        }

    @staticmethod
    def scenario_summary(
        bonds: list,
        base,
        magnitude_bps: float = 100.0,
    ) -> dict[str, dict]:
        """
        Compute PnL for each scenario across a list of FixedRateBond objects.
        Returns {scenario_name: {bond_label: pnl, 'total': pnl}}.
        """
        base_prices = {i: b.price(base) for i, b in enumerate(bonds)}
        results = {}

        for shock_type in ShockType:
            shocked = ShockEngine.apply(base, shock_type, magnitude_bps)
            pnl = {}
            for i, b in enumerate(bonds):
                px_shocked = b.price(shocked)
                pnl[f"bond_{i}"] = px_shocked - base_prices[i]
            pnl["total"] = sum(pnl.values())
            results[shock_type.value] = pnl

        return results
