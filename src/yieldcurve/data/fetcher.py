"""FRED Treasury data fetcher."""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[4] / ".env")

# FRED series ID → maturity in years
TREASURY_SERIES: dict[str, float] = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 3 / 12,
    "DGS6MO": 6 / 12,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}

_MATURITY_LABEL: dict[str, str] = {
    "DGS1MO": "1M",
    "DGS3MO": "3M",
    "DGS6MO": "6M",
    "DGS1": "1Y",
    "DGS2": "2Y",
    "DGS3": "3Y",
    "DGS5": "5Y",
    "DGS7": "7Y",
    "DGS10": "10Y",
    "DGS20": "20Y",
    "DGS30": "30Y",
}


def fetch_treasury_yields(
    start_date: str | date | datetime = "2000-01-01",
    end_date: str | date | datetime | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch constant-maturity Treasury yields from FRED.

    Returns a DataFrame indexed by date with columns named by label (e.g. '1M', '2Y').
    Yields are in percent (as FRED provides them), NOT decimals.
    Missing observations (holidays/weekends) are forward-filled then dropped if still NaN.
    """
    try:
        from fredapi import Fred
    except ImportError as e:
        raise ImportError("pip install fredapi") from e

    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError(
            "Provide api_key= or set FRED_API_KEY in your .env file. "
            "Free key at https://fred.stlouisfed.org/docs/api/fred/"
        )

    fred = Fred(api_key=key)
    end = end_date or date.today()

    series: dict[str, pd.Series] = {}
    for fred_id, label in _MATURITY_LABEL.items():
        try:
            s = fred.get_series(fred_id, observation_start=start_date, observation_end=end)
            s.name = label
            series[label] = s
        except Exception as exc:
            print(f"Warning: could not fetch {fred_id}: {exc}")

    df = pd.DataFrame(series)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.ffill().dropna(how="all")
    return df


def yields_for_date(
    df: pd.DataFrame,
    as_of: str | date | datetime | None = None,
) -> pd.Series:
    """
    Extract one row from a yields DataFrame.
    If *as_of* is None, returns the most recent row.
    Falls back to the nearest prior business day if exact date is missing.
    """
    target = pd.Timestamp(as_of) if as_of is not None else df.index[-1]
    if target in df.index:
        row = df.loc[target]
    else:
        prior = df.index[df.index <= target]
        if prior.empty:
            raise ValueError(f"No data on or before {target}")
        row = df.loc[prior[-1]]

    row = row.dropna()
    # Convert percent → decimal
    return row / 100.0


def maturity_years_for(labels: list[str]) -> list[float]:
    """Return maturity in years for a list of column labels like ['1M', '2Y']."""
    label_to_years = {v: k_years for (k_id, k_years), v in
                      zip(TREASURY_SERIES.items(), _MATURITY_LABEL.values())}
    return [label_to_years[lbl] for lbl in labels]
