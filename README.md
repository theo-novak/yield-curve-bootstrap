# Yield Curve Explorer

Bootstrap zero-coupon term structures from US Treasury data, compare interpolation methods, price fixed-income instruments, and stress-test with rate shock scenarios.

## Stack

Python · pandas · numpy · scipy · DuckDB · Plotly · Streamlit

## Project layout

```text
src/yieldcurve/
├── data/
│   ├── fetcher.py       FRED API client — pulls DGS* Treasury CMT series
│   └── storage.py       DuckDB cache (date × tenor → yield)
├── curve/
│   ├── bootstrap.py     Iterative bootstrap: par yields → zero rates + DFs
│   ├── interpolation.py Linear, cubic spline, log-linear, cubic log-DF methods
│   └── nelson_siegel.py Nelson-Siegel and NSS parametric fit (global optimiser)
├── risk/
│   ├── instruments.py   FixedRateBond: price, YTM, duration, DV01, convexity
│   └── scenarios.py     Parallel, steepener, flattener, twist shocks; key-rate DV01
└── viz/
    └── plots.py         Reusable Plotly figures (zero curve, forward, surface, …)

dashboard/app.py          Streamlit dashboard
notebooks/
└── 01_bootstrap_walkthrough.ipynb   Step-by-step walkthrough with formulas
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Get a free FRED API key
#    → https://fred.stlouisfed.org/docs/api/fred/
cp .env.example .env
# Edit .env and set FRED_API_KEY=<your key>
```

## Fetch data

```python
from yieldcurve.data.fetcher import fetch_treasury_yields
from yieldcurve.data.storage import save_yields

df = fetch_treasury_yields("2000-01-01", "2024-12-31")
save_yields(df)   # cached in data/yields.duckdb
```

## Bootstrap a curve

```python
from yieldcurve.data.fetcher import yields_for_date
from yieldcurve.data.storage import load_yields
from yieldcurve.curve.bootstrap import BootstrappedCurve
from yieldcurve.curve.interpolation import InterpolatedCurve, InterpolationMethod

par = yields_for_date(load_yields(), "2024-01-02")   # returns decimal values
boot = BootstrappedCurve.from_series(par)
curve = InterpolatedCurve(boot, InterpolationMethod.CUBIC_LOG_DF)

print(f"10Y zero rate:    {curve.zero_rate(10)*100:.4f}%")
print(f"10Y discount DF:  {curve.discount_factor(10):.6f}")
print(f"5Y→10Y fwd rate:  {curve.forward_rate(5, 10)*100:.4f}%")
```

## Bond analytics

```python
from yieldcurve.risk.instruments import FixedRateBond

bond = FixedRateBond(coupon_rate=0.04, maturity=10)
report = bond.risk_report(curve)
# price, ytm, macaulay_duration, modified_duration, dv01, convexity
```

## Shock scenarios

```python
from yieldcurve.risk.scenarios import ShockEngine, ShockType

# Named scenarios at ±100 bp default magnitude
parallel_up  = ShockEngine.apply(curve, ShockType.PARALLEL_UP, 100)
steepener    = ShockEngine.apply(curve, ShockType.BEAR_STEEPENER, 100)
all_shocks   = ShockEngine.all_scenarios(curve, 100)

# Single key-rate bump (for DV01 ladder)
kr_10y = ShockEngine.key_rate_shift(curve, "10Y", shift_bps=1.0)
```

## Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard includes:

- **Zero Curve tab** — bootstrapped curve, forward rates, discount factors, Nelson-Siegel fit
- **Interpolation tab** — side-by-side comparison of all four methods + their forward curves
- **Shock Scenarios tab** — interactive shock selector with scenario rate tables
- **Bond Analytics tab** — bond pricer, P&L under shocks, key-rate DV01 ladder
- **History tab** — yield history and 10Y–2Y spread chart
- **3-D Surface tab** — yield surface over time

## Bootstrap methodology

US Treasury CMT par yields are bootstrapped as follows:

**T-bills (T ≤ 6M):** already zero-coupon discount instruments.
$$DF(T) = \frac{1}{1 + c \cdot T}$$

**Notes/Bonds (T ≥ 1Y):** semiannual coupon bonds priced at par. Solve iteratively for each maturity T:
$$100 = \frac{c}{2} \cdot 100 \sum_{i=1}^{n-1} DF(t_i) + 100 \left(1 + \frac{c}{2}\right) DF(T)$$
$$\Rightarrow DF(T) = \frac{100 - \frac{c}{2} \cdot 100 \sum_{i<n} DF(t_i)}{100 \cdot (1 + c/2)}$$

Intermediate coupon dates not at bootstrap nodes use log-linear interpolation between known discount factors.

Annually compounded zero rate: $z(T) = DF(T)^{-1/T} - 1$

## Interpolation methods

| Method | Description | Forward curve smoothness |
| --- | --- | --- |
| `linear_zero` | Linear on zero rates | Piecewise constant (step) |
| `cubic_zero` | Natural cubic spline on zero rates | Smooth but may oscillate |
| `log_linear_df` | Log-linear on discount factors | Flat-forward, piecewise const |
| `cubic_log_df` | Cubic spline on log-DF | Smooth spot + smooth fwd (**recommended**) |

## Rate shock scenarios

| Scenario | Short end | Long end | Economic interpretation |
| --- | --- | --- | --- |
| Parallel ±100 bp | ±100 | ±100 | General rate level shift |
| Bear steepener | +150 | +50 | Policy tightening, short rates spike |
| Bull steepener | -150 | -50 | Fed cuts expected, short rates fall |
| Bear flattener | +50 | +150 | Term premium rise, recession risk |
| Bull flattener | -50 | -150 | Growth fear, long rates rally |
| Twist | +75 | -75 | Butterfly: 2Y up, 10Y down |
