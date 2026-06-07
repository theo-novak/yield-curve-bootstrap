"""
Nelson-Siegel and Nelson-Siegel-Svensson parametric curve models.

Nelson-Siegel (4 params):
    r(t) = β₀ + (β₁ + β₂) * (λ/t) * (1 - exp(-t/λ)) - β₂ * exp(-t/λ)

Nelson-Siegel-Svensson (6 params):
    r(t) = β₀ + β₁ * (λ₁/t)*(1-exp(-t/λ₁))
               + β₂ * [(λ₁/t)*(1-exp(-t/λ₁)) - exp(-t/λ₁)]
               + β₃ * [(λ₂/t)*(1-exp(-t/λ₂)) - exp(-t/λ₂)]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, differential_evolution


@dataclass
class NelsonSiegelParams:
    beta0: float
    beta1: float
    beta2: float
    lam: float
    rmse_bps: float = 0.0


@dataclass
class NSSParams:
    beta0: float
    beta1: float
    beta2: float
    beta3: float
    lam1: float
    lam2: float
    rmse_bps: float = 0.0


class NelsonSiegelCurve:
    """Fitted Nelson-Siegel parametric yield curve."""

    def __init__(self, params: NelsonSiegelParams) -> None:
        self.params = params

    @classmethod
    def fit(
        cls,
        tenors: np.ndarray,
        zero_rates: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> "NelsonSiegelCurve":
        """
        Fit NS model to observed (tenor, zero_rate) pairs.
        Uses differential evolution for global search then Nelder-Mead polish.
        """
        tenors = np.asarray(tenors, dtype=float)
        zero_rates = np.asarray(zero_rates, dtype=float)
        w = np.ones_like(tenors) if weights is None else np.asarray(weights, dtype=float)

        def objective(p):
            b0, b1, b2, lam = p
            if lam <= 0:
                return 1e10
            fitted = _ns_rate(tenors, b0, b1, b2, lam)
            return float(np.sum(w * (fitted - zero_rates) ** 2))

        bounds = [
            (0.0, 0.20),   # β₀: long-term level
            (-0.15, 0.15), # β₁: short-term slope
            (-0.15, 0.15), # β₂: curvature
            (0.1, 5.0),    # λ: decay (years)
        ]

        de_result = differential_evolution(objective, bounds, seed=42, maxiter=500, tol=1e-10)
        result = minimize(objective, de_result.x, method="Nelder-Mead",
                          options={"maxiter": 10_000, "xatol": 1e-10, "fatol": 1e-12})

        b0, b1, b2, lam = result.x
        fitted = _ns_rate(tenors, b0, b1, b2, lam)
        rmse = float(np.sqrt(np.mean((fitted - zero_rates) ** 2))) * 10_000  # bps

        return cls(NelsonSiegelParams(b0, b1, b2, lam, rmse))

    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        scalar = np.isscalar(t)
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        p = self.params
        result = _ns_rate(t_arr, p.beta0, p.beta1, p.beta2, p.lam)
        return float(result[0]) if scalar else result

    def discount_factor(self, t: float | np.ndarray) -> float | np.ndarray:
        z = self.zero_rate(t)
        t_arr = np.asarray(t, dtype=float)
        return (1.0 + z) ** (-t_arr)

    def forward_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """Instantaneous (continuous) forward rate at t."""
        scalar = np.isscalar(t)
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        p = self.params
        result = _ns_forward(t_arr, p.beta0, p.beta1, p.beta2, p.lam)
        return float(result[0]) if scalar else result

    def parameter_table(self) -> dict[str, float]:
        p = self.params
        return {
            "β₀ (level)": p.beta0,
            "β₁ (slope)": p.beta1,
            "β₂ (curvature)": p.beta2,
            "λ (decay yrs)": p.lam,
            "RMSE (bps)": p.rmse_bps,
        }


class NelsonSiegelSvenssonCurve:
    """Fitted Nelson-Siegel-Svensson curve (6 parameters)."""

    def __init__(self, params: NSSParams) -> None:
        self.params = params

    @classmethod
    def fit(
        cls,
        tenors: np.ndarray,
        zero_rates: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> "NelsonSiegelSvenssonCurve":
        tenors = np.asarray(tenors, dtype=float)
        zero_rates = np.asarray(zero_rates, dtype=float)
        w = np.ones_like(tenors) if weights is None else np.asarray(weights, dtype=float)

        def objective(p):
            b0, b1, b2, b3, l1, l2 = p
            if l1 <= 0 or l2 <= 0 or abs(l1 - l2) < 0.05:
                return 1e10
            fitted = _nss_rate(tenors, b0, b1, b2, b3, l1, l2)
            return float(np.sum(w * (fitted - zero_rates) ** 2))

        bounds = [
            (0.0, 0.20), (-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15),
            (0.1, 5.0), (0.1, 5.0),
        ]

        de_result = differential_evolution(objective, bounds, seed=42, maxiter=800, tol=1e-10)
        result = minimize(objective, de_result.x, method="Nelder-Mead",
                          options={"maxiter": 15_000, "xatol": 1e-10, "fatol": 1e-12})

        b0, b1, b2, b3, l1, l2 = result.x
        fitted = _nss_rate(tenors, b0, b1, b2, b3, l1, l2)
        rmse = float(np.sqrt(np.mean((fitted - zero_rates) ** 2))) * 10_000

        return cls(NSSParams(b0, b1, b2, b3, l1, l2, rmse))

    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        scalar = np.isscalar(t)
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        p = self.params
        result = _nss_rate(t_arr, p.beta0, p.beta1, p.beta2, p.beta3, p.lam1, p.lam2)
        return float(result[0]) if scalar else result

    def discount_factor(self, t: float | np.ndarray) -> float | np.ndarray:
        z = self.zero_rate(t)
        return (1.0 + np.asarray(z)) ** (-np.asarray(t, dtype=float))


# ------------------------------------------------------------------
# Pure functions
# ------------------------------------------------------------------

def _ns_rate(t: np.ndarray, b0, b1, b2, lam) -> np.ndarray:
    t = np.where(t < 1e-8, 1e-8, t)
    decay = np.exp(-t / lam)
    loading = (lam / t) * (1.0 - decay)
    return b0 + (b1 + b2) * loading - b2 * decay


def _ns_forward(t: np.ndarray, b0, b1, b2, lam) -> np.ndarray:
    """Instantaneous forward rate for Nelson-Siegel."""
    t = np.where(t < 1e-8, 1e-8, t)
    decay = np.exp(-t / lam)
    return b0 + b1 * decay + b2 * (t / lam) * decay


def _nss_rate(t: np.ndarray, b0, b1, b2, b3, l1, l2) -> np.ndarray:
    t = np.where(t < 1e-8, 1e-8, t)
    d1 = np.exp(-t / l1)
    d2 = np.exp(-t / l2)
    term1 = b1 * (l1 / t) * (1.0 - d1)
    term2 = b2 * ((l1 / t) * (1.0 - d1) - d1)
    term3 = b3 * ((l2 / t) * (1.0 - d2) - d2)
    return b0 + term1 + term2 + term3
