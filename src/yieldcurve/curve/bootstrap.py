"""
Zero-coupon curve bootstrapping from CMT par yields.

US Treasury Constant Maturity (CMT) rates from FRED are par yields:
  - Bills (≤ 6M):   zero-coupon discount instruments; spot rate = CMT rate
  - Notes/Bonds (≥ 1Y): semiannual coupon bonds priced at par; bootstrap iteratively

Bootstrap procedure
-------------------
For each maturity T (sorted ascending):
  1. If T ≤ 0.5: DF(T) = 1 / (1 + c * T)   where c = CMT rate
  2. If T ≥ 1.0: the CMT bond pays c/2 per period semiannually at par=100.
     Known coupon dates d₁ … d_{n-1} use already-computed DF values
     (interpolated if not a node). Unknown DF(T) solved from:
       100 = (c/2)*100 * Σ DF(dᵢ) + 100 * DF(T)
       → DF(T) = (100 - (c/2)*100 * Σ_{i<n} DF(dᵢ)) / (100*(1 + c/2))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class BootstrappedCurve:
    """Bootstrapped zero-coupon curve built from CMT par yields."""

    tenors: np.ndarray          # maturity in years, sorted
    zero_rates: np.ndarray      # annually compounded zero rates (decimal)
    discount_factors: np.ndarray
    curve_date: pd.Timestamp | None = None

    # Internal log-DF cache used during construction (not serialised)
    _df_cache: dict[float, float] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_par_yields(
        cls,
        par_yields: dict[float, float],
        curve_date: pd.Timestamp | None = None,
    ) -> "BootstrappedCurve":
        """
        par_yields: {maturity_years: par_yield_decimal}
                    e.g. {0.25: 0.0515, 0.5: 0.0525, 1.0: 0.0530, ...}
        """
        sorted_items = sorted(par_yields.items())
        tenors: list[float] = []
        zero_rates: list[float] = []
        discount_factors: list[float] = []
        df_cache: dict[float, float] = {}

        for T, c in sorted_items:
            if T <= 6 / 12:
                # T-bill: simple-interest discount security
                df = 1.0 / (1.0 + c * T)
            else:
                # Coupon bond: semiannual, at par → solve for DF(T)
                coupon_dates = np.round(np.arange(0.5, T + 1e-9, 0.5), 8)
                coupon_cash = c / 2 * 100  # per $100 par

                pv_known = sum(
                    coupon_cash * _interp_df(d, df_cache)
                    for d in coupon_dates[:-1]
                )
                df = (100.0 - pv_known) / (100.0 * (1.0 + c / 2.0))

            df_cache[T] = df
            z = df ** (-1.0 / T) - 1.0  # annually compounded zero rate

            tenors.append(T)
            zero_rates.append(z)
            discount_factors.append(df)

        obj = cls(
            tenors=np.array(tenors),
            zero_rates=np.array(zero_rates),
            discount_factors=np.array(discount_factors),
            curve_date=curve_date,
        )
        obj._df_cache = df_cache
        return obj

    @classmethod
    def from_series(
        cls,
        yields: pd.Series,
        curve_date: pd.Timestamp | None = None,
    ) -> "BootstrappedCurve":
        """
        Convenience constructor from a pd.Series with label index (e.g. '1M', '2Y')
        and decimal yield values.
        """
        label_to_years = {
            "1M": 1 / 12, "3M": 3 / 12, "6M": 6 / 12,
            "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "5Y": 5.0,
            "7Y": 7.0, "10Y": 10.0, "20Y": 20.0, "30Y": 30.0,
        }
        par_yields = {
            label_to_years[lbl]: val
            for lbl, val in yields.items()
            if lbl in label_to_years
        }
        return cls.from_par_yields(par_yields, curve_date=curve_date)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def discount_factor(self, t: float | np.ndarray) -> float | np.ndarray:
        """Log-linearly interpolated discount factor."""
        return np.exp(np.interp(
            np.asarray(t, dtype=float),
            self.tenors,
            np.log(self.discount_factors),
        ))

    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """Annually compounded zero rate from the bootstrapped nodes (linear interp)."""
        df = self.discount_factor(t)
        t_arr = np.asarray(t, dtype=float)
        return df ** (-1.0 / t_arr) - 1.0

    def continuous_zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        df = self.discount_factor(t)
        return -np.log(df) / np.asarray(t, dtype=float)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Continuously compounded forward rate between t1 and t2."""
        if t2 <= t1:
            raise ValueError("t2 must be > t1")
        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        return -np.log(df2 / df1) / (t2 - t1)

    def forward_curve(self, step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
        """Instantaneous forward rates on a fine grid."""
        t_grid = np.arange(step, self.tenors[-1] + step / 2, step)
        fwd = np.array([self.forward_rate(max(t - step, 1e-6), t) for t in t_grid])
        return t_grid, fwd

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "tenor": self.tenors,
                "zero_rate": self.zero_rates,
                "discount_factor": self.discount_factors,
                "continuous_rate": -np.log(self.discount_factors) / self.tenors,
            }
        )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _interp_df(t: float, df_cache: dict[float, float]) -> float:
    """
    Log-linearly interpolate DF at t from df_cache.
    t must be ≤ max(df_cache) — called only for coupon dates strictly inside
    the already-bootstrapped region.
    """
    if t in df_cache:
        return df_cache[t]

    known = sorted(df_cache.keys())
    if t <= known[0]:
        # Extrapolate flat (short end)
        return df_cache[known[0]] ** (t / known[0])

    t_lo = max(k for k in known if k <= t)
    candidates_hi = [k for k in known if k > t]
    if not candidates_hi:
        # Beyond current longest node — should not happen during bootstrap
        t_hi = t_lo
        return df_cache[t_hi]
    t_hi = min(candidates_hi)

    alpha = (t - t_lo) / (t_hi - t_lo)
    log_df = (1 - alpha) * np.log(df_cache[t_lo]) + alpha * np.log(df_cache[t_hi])
    return float(np.exp(log_df))
