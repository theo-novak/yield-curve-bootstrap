"""
Interpolation methods for zero-coupon yield curves.

Four methods implemented:
  LINEAR_ZERO       — linear interpolation on zero rates
  CUBIC_ZERO        — natural cubic spline on zero rates
  LOG_LINEAR_DF     — log-linear interpolation on discount factors (piecewise const fwd)
  CUBIC_LOG_DF      — cubic spline on log-discount-factors (smooth fwd rates)
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

from .bootstrap import BootstrappedCurve


class InterpolationMethod(str, Enum):
    LINEAR_ZERO = "linear_zero"
    CUBIC_ZERO = "cubic_zero"
    LOG_LINEAR_DF = "log_linear_df"
    CUBIC_LOG_DF = "cubic_log_df"


class InterpolatedCurve:
    """
    Wraps a BootstrappedCurve and adds smooth interpolation over a dense grid
    and/or arbitrary query tenors, using a user-selected method.
    """

    def __init__(
        self,
        base: BootstrappedCurve,
        method: InterpolationMethod = InterpolationMethod.CUBIC_LOG_DF,
    ) -> None:
        self.base = base
        self.method = method
        self._build_interpolant()

    def _build_interpolant(self) -> None:
        t = self.base.tenors
        z = self.base.zero_rates
        log_df = np.log(self.base.discount_factors)

        m = self.method
        if m == InterpolationMethod.LINEAR_ZERO:
            self._interp = interp1d(t, z, kind="linear", fill_value="extrapolate")
            self._query = lambda ts: self._interp(ts)
            self._mode = "zero"

        elif m == InterpolationMethod.CUBIC_ZERO:
            cs = CubicSpline(t, z, bc_type="not-a-knot", extrapolate=True)
            self._query = lambda ts: cs(ts)
            self._mode = "zero"

        elif m == InterpolationMethod.LOG_LINEAR_DF:
            self._interp = interp1d(t, log_df, kind="linear", fill_value="extrapolate")
            self._query = lambda ts: np.exp(self._interp(ts))
            self._mode = "df"

        elif m == InterpolationMethod.CUBIC_LOG_DF:
            cs = CubicSpline(t, log_df, bc_type="not-a-knot", extrapolate=True)
            self._query = lambda ts: np.exp(cs(ts))
            self._mode = "df"

    def discount_factor(self, t: float | np.ndarray) -> float | np.ndarray:
        scalar = np.isscalar(t)
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))

        if self._mode == "df":
            result = self._query(t_arr)
        else:
            z = self._query(t_arr)
            result = (1.0 + z) ** (-t_arr)

        return float(result[0]) if scalar else result

    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        scalar = np.isscalar(t)
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))

        if self._mode == "zero":
            result = self._query(t_arr)
        else:
            df = self._query(t_arr)
            result = df ** (-1.0 / t_arr) - 1.0

        return float(result[0]) if scalar else result

    def continuous_zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        df = self.discount_factor(t)
        t_arr = np.asarray(t, dtype=float)
        return -np.log(df) / t_arr

    def forward_rate(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            raise ValueError("t2 > t1 required")
        df1 = float(self.discount_factor(t1))
        df2 = float(self.discount_factor(t2))
        return -np.log(df2 / df1) / (t2 - t1)

    def forward_curve(self, step: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        t_max = self.base.tenors[-1]
        t_grid = np.arange(step, t_max + step / 2, step)
        fwd = np.array([self.forward_rate(max(t - step, 1e-6), t) for t in t_grid])
        return t_grid, fwd

    def dense_grid(self, n: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (tenors, zero_rates, discount_factors) on a fine grid."""
        t_grid = np.linspace(self.base.tenors[0], self.base.tenors[-1], n)
        z = self.zero_rate(t_grid)
        df = self.discount_factor(t_grid)
        return t_grid, z, df

    @property
    def curve_date(self):
        return self.base.curve_date
