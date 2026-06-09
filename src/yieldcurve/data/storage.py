"""DuckDB-backed local cache for Treasury yields."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

_DEFAULT_DB = Path(__file__).parents[4] / "data" / "yields.duckdb"

_DDL = """
CREATE TABLE IF NOT EXISTS treasury_yields (
    date        DATE NOT NULL,
    label       VARCHAR NOT NULL,  -- e.g. '10Y'
    yield_pct   DOUBLE NOT NULL,   -- in percent, as FRED provides
    PRIMARY KEY (date, label)
);
"""


def get_db(path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    db_path = Path(path) if path else _DEFAULT_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute(_DDL)
    return conn


def save_yields(df: pd.DataFrame, path: str | Path | None = None) -> None:
    """
    Persist a yields DataFrame (date-indexed, label-column, percent values).
    Upserts — safe to call multiple times on overlapping date ranges.
    """
    conn = get_db(path)
    long = df.reset_index().melt(id_vars="date", var_name="label", value_name="yield_pct")
    long = long.dropna(subset=["yield_pct"])
    conn.execute(
        "INSERT OR REPLACE INTO treasury_yields (date, label, yield_pct) SELECT date, label, yield_pct FROM long"
    )
    conn.close()


def load_yields(
    start_date: str | None = None,
    end_date: str | None = None,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load cached yields as a wide DataFrame (date × label).
    Returns an empty DataFrame if no data is found.
    """
    conn = get_db(path)
    where_parts = []
    params: list[str] = []
    if start_date:
        where_parts.append("date >= ?")
        params.append(start_date)
    if end_date:
        where_parts.append("date <= ?")
        params.append(end_date)

    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
    query = f"SELECT date, label, yield_pct FROM treasury_yields {where_clause} ORDER BY date, label"
    rows = conn.execute(query, params).fetchdf()
    conn.close()

    if rows.empty:
        return pd.DataFrame()

    wide = rows.pivot(index="date", columns="label", values="yield_pct")
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "date"

    ordered = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    cols = [c for c in ordered if c in wide.columns]
    return wide[cols]


def cached_date_range(path: str | Path | None = None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    conn = get_db(path)
    row = conn.execute("SELECT MIN(date), MAX(date) FROM treasury_yields").fetchone()
    conn.close()
    if row is None or row[0] is None:
        return None, None
    return pd.Timestamp(row[0]), pd.Timestamp(row[1])
