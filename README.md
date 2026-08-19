# Arbitrage-Free Implied-Volatility Surface Construction and Calibration

A volatility surface can fit the market beautifully and still be impossible — implying negative
probabilities, or that a longer-dated option is worth less than a shorter-dated one covering the
same outcomes. This study builds a surface anchored to observed market data, tests it against the
conditions that rule both out, and measures what enforcing them costs.

<!-- Once this repository is on GitHub, replace OWNER/REPO below to activate the CI badge:
[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
-->
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230)](https://docs.astral.sh/ruff/)
[![Types: mypy strict](https://img.shields.io/badge/types-mypy%20strict-1f5082)](https://mypy-lang.org/)

## Three findings

**1. The published market curve never breached the calendar condition — not once in fifteen years.**
Across 3,891 trading days from January 2011 to June 2026, total implied variance was non-decreasing
in maturity on every single day. The smallest forward variance observed anywhere was **+0.0092**,
comfortably positive. That covers Volmageddon, the Covid crash, the 2022 rate shock, the yen carry
unwind and the 2025 tariff shock.

**2. An inverted volatility curve is not an arbitrage — and the distinction is routinely muddled.**
On **395 days (10.2%)** short-dated implied volatility sat above long-dated, the familiar sign of
market stress. **Every one of those days was still arbitrage-free.** Total variance carries a factor
of maturity, so a curve can invert in volatility while total variance keeps rising. Calling a VIX
inversion an arbitrage confuses a stress signal with a violated constraint.

**3. Fitting observed skew without a constraint produces impossible surfaces a fifth of the time.**
Calibrated to reproduce the risk-neutral skewness the market prices, the surface implies a negative
probability density on **21% of days**. Imposing the no-arbitrage conditions fixes that completely —
100% valid — at the cost of a nine-fold worse fit to observed skew.

| | Fit to observed skew (RMSE) | Days free of arbitrage | Sufficient conditions hold | Worst g(k) |
|---|---:|---:|---:|---:|
| Unconstrained | **0.067** | 78.9% | 0% | **−3.15** |
| No-arbitrage enforced | 0.633 | **100%** | **100%** | **+0.25** |

That trade-off is the study's central result. It is not a tuning artefact: it is the price of
insisting that a surface describe a distribution that could actually exist.

## Read this first: what is and is not observed

| Component | Status |
|---|---|
| Maturity dimension | **Observed.** Cboe publishes at-the-money expected volatility at six horizons |
| Calendar no-arbitrage test | **Run directly on that published curve** — no model involved |
| Strike dimension | **Not observed.** No free source gives implied volatility strike by strike |
| Skew calibration target | **Observed, but one number per day** — the skewness encoded in Cboe's SKEW index |
| No-arbitrage conditions | **Theorems.** True of the parameterisation regardless of any data |

The honest position: finding 1 and finding 2 rest entirely on published market data and involve no
model at all. Finding 3 is a statement about calibrating a specific surface family to a specific
(thin) target — one skewness per day is a far weaker constraint than a quoted option chain, and the
fit statistics should be read in that light. What holds regardless of data quality are the
no-arbitrage conditions themselves, which are mathematical results, verified numerically here.

## The observed term structure

| Index | Horizon | Mean IV | 10th–90th percentile | Mean total variance |
|---|---:|---:|---:|---:|
| VIX9D | 9 days | 17.5% | 10.8% – 26.9% | 0.00092 |
| VIX | 30 days | 18.2% | 12.3% – 26.4% | 0.00310 |
| VIX3M | 93 days | 20.1% | 14.5% – 28.2% | 0.01119 |
| VIX6M | 186 days | 21.5% | 16.2% – 29.4% | 0.02508 |
| VIX1Y | 365 days | 22.7% | 17.6% – 29.6% | 0.05340 |

Volatility rises with horizon and total variance rises much faster, because it multiplies by
maturity. The right-hand column is the one no-arbitrage constrains.

### The held-out maturity test

A smooth three-parameter variance curve is fitted to four of the five maturities, then asked to
predict the fifth — a maturity it never saw:

| | |
|---|---:|
| Median absolute error | **1.82%** |
| Within 1% | 27.8% |
| Within 5% | **92.1%** |
| 99th percentile | 10.7% |
| Worst day | 28.3% |

The term structure is smooth enough that four points pin the fifth to within a couple of percent on
a typical day. The tail belongs to days when the front of the curve moved violently.

## Where the constrained surface breaks down

The constrained surface matches the market's observed skewness **exactly on 71% of days**. On the
other 29% it saturates against the no-arbitrage bound and cannot reach the target.

Those days are not random. On days it matches, the market priced an average skewness of **−2.59**;
on the days it cannot, **−4.70**, with the most extreme unreachable target at **−8.31**.

In other words: on the days the market prices its most violently negative skew, no surface in this
family can reproduce that skew and remain arbitrage-free. The reading is not that the market is
arbitrageable — the calendar evidence says otherwise — but that SSVI's single global shape function
is too rigid for those days. A production surface would free more parameters per slice.

## Stress episodes

| Episode | Date | Peak 30-day IV | Peak SKEW | Most negative skewness | Calendar violations | Days inverted |
|---|---|---:|---:|---:|---:|---:|
| Volmageddon | 2018-02-05 | 37.3% | 148.0 | −4.80 | **0** | 8 |
| Covid crash | 2020-03-16 | 82.7% | 132.4 | −3.24 | **0** | 15 |
| Rate shock | 2022-06-13 | 34.0% | 125.2 | −2.52 | **0** | 4 |
| Yen carry unwind | 2024-08-05 | 38.6% | 147.4 | −4.74 | **0** | 6 |
| Tariff shock | 2025-04-04 | 52.3% | 148.7 | −4.87 | **0** | 11 |

Note that the Covid crash produced the highest volatility but *not* the most negative skew — SKEW
peaked higher in 2018, 2024 and 2025. When everything is already priced for disaster, the extra
premium for a crash narrows.

## Method

### Coordinates

Everything is in log-moneyness `k = ln(K/F)` and **total implied variance** `w(k,T) = IV(k,T)² · T`.
Total variance is the natural coordinate because the no-arbitrage conditions are statements about
it.

### The two parameterisations

**Raw SVI** fits one maturity slice:

```text
w(k) = a + b · [ rho·(k - m) + sqrt((k - m)² + sigma²) ]
```

**SSVI** fits the whole surface, tying every slice to an at-the-money variance curve `theta(T)`:

```text
w(k, theta) = theta/2 · { 1 + rho·phi(theta)·k + sqrt([phi(theta)·k + rho]² + 1 - rho²) }
```

SSVI is used here for a practical reason: `theta(T)` is *observable*. Cboe publishes it, so the
surface's backbone is market data rather than a fitted guess. At `k = 0` the formula collapses to
`theta` exactly.

### The conditions

- **Butterfly.** The implied density must be non-negative everywhere; the test is the sign of
  Gatheral's `g(k)`. Where it goes negative, a butterfly spread has a negative price.
- **Calendar.** Total variance must not fall with maturity.

Gatheral and Jacquier give *sufficient* conditions on the SSVI parameters directly:

```text
theta·phi(theta)·(1 + |rho|) < 4        and        theta·phi(theta)²·(1 + |rho|) <= 4
```

Rearranging them for `|rho|` turns a test into a **bound**, which is what makes constrained
calibration possible: the fit is confined to the feasible region rather than checked afterwards.
Feasibility is then guaranteed, not encouraged — after fitting, the surface is projected into the
region by bisection on the skew scale, so the returned surface always satisfies the conditions.

Being sufficient and not necessary, the conditions can fail on a surface whose density is
nonetheless perfectly valid. The test suite keeps a fixture of exactly that case, so the parameter
test and the numerical scan are never quietly assumed to be the same test.

## Verification

The mathematical claims are checked against closed forms and against an independent code path:

- A flat slice reproduces the **lognormal density exactly** (agreement to 4×10⁻¹⁶) and `g(k) ≡ 1`.
- Every SSVI slice equals its raw-SVI representation to machine precision (3×10⁻¹⁷).
- The implied density **integrates to 1** to within 7×10⁻⁷ at every maturity.
- The density matches the one recovered from **option prices by Breeden–Litzenberger** — the second
  derivative of the call price in strike — computed through the Black-Scholes pricer, an entirely
  separate code path. Agreement is corroboration, not restatement.
- A deliberately arbitrageable surface is caught by both detectors.

```bash
make lint       # ruff check + format check
make typecheck  # mypy, strict
make test       # 217 tests
make verify     # re-run and diff against outputs/
```

The fast suite runs in about 90 seconds. The end-to-end reproduction check is slower — it
calibrates every trading day under two regimes, which takes several minutes — so it carries its own
marker and is excluded from `make test-fast`.

## Quickstart

Requires Python 3.11 or newer; developed and validated on 3.13.

```bash
make setup      # install Python 3.13 and dependencies
make data       # download the Cboe index histories
make reproduce  # run the study, writing to outputs/
```

Or with plain pip:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/fetch_cboe_data.py
ivsurface --output-dir outputs
```

A successful run prints:

```text
Completed 3891 days: 0 calendar violations in the observed curve, held-out maturity predicted to 1.82% median error, and enforcing no-arbitrage moved the surface from 78.9% to 100.0% valid at a skewness RMSE of 0.067 against 0.633.
```

The surface is importable too:

```python
import numpy as np
from ivsurface import SSVI
from ivsurface.diagnostics import check_surface

surface = SSVI(rho=-0.59, eta=1.13, gamma=0.54)
print(surface.total_variance(np.linspace(-0.3, 0.3, 7), theta=0.031))
print(check_surface(surface, [0.0009, 0.0031, 0.0112, 0.0251, 0.0534]).arbitrage_free)
```

## How the code is organised

| Module | Responsibility |
|---|---|
| [`svi.py`](src/ivsurface/svi.py) | The parameterisations, the density, and the no-arbitrage conditions |
| [`termstructure.py`](src/ivsurface/termstructure.py) | The observed variance curve and the calendar test |
| [`calibrate.py`](src/ivsurface/calibrate.py) | Variance-curve fitting, held-out testing, constrained SSVI fitting |
| [`diagnostics.py`](src/ivsurface/diagnostics.py) | Butterfly scans, density mass, per-day diagnosis |
| [`config.py`](src/ivsurface/config.py) | Study period, grids, tolerances, the data contract |
| [`loaders.py`](src/ivsurface/loaders.py) | Reading Cboe history, failing loudly |
| [`aggregate.py`](src/ivsurface/aggregate.py) | Summaries, episodes, stability, bootstrap |
| [`figures.py`](src/ivsurface/figures.py) | The four figures |
| [`verify.py`](src/ivsurface/verify.py) | Tolerance-based comparison of two result sets |
| [`pipeline.py`](src/ivsurface/pipeline.py) | End-to-end orchestration |
| [`cli.py`](src/ivsurface/cli.py) | Command-line interface |

## Reproducibility

`outputs/` holds the committed result set and the test suite checks the study still produces it.

**The study period ends on a fixed date on purpose.** Cboe extends these series every trading day,
so an open-ended sample would answer differently on every download. Freezing the end is what lets a
download taken months later reproduce the published numbers.

One numerical detail worth recording, because getting it wrong misreports a headline check: the
density integration grid is built in two parts. A uniform grid cannot be both wide enough for SVI's
fat wings and fine enough to resolve a nine-day slice — whose standard deviation in log-moneyness
is about 0.03 — without becoming enormous. A dense core with sparse wings is accurate to 7×10⁻⁷ at
a third the size. A coarser uniform grid silently reported density masses off by 7%.

## Data provenance

Cboe's index history is not redistributed here; `scripts/fetch_cboe_data.py` downloads it from
<https://cdn.cboe.com/api/global/us_indices/daily_prices/>: VIX1D, VIX9D, VIX, VIX3M, VIX6M, VIX1Y,
SKEW and VVIX.

Two loading details the code handles explicitly. Cboe publishes these files in two shapes — some
carry `DATE,OPEN,HIGH,LOW,CLOSE`, others `DATE` plus a column named after the index — so the value
column is resolved rather than assumed. And SKEW is an index near 100 encoding a skewness through
`SKEW = 100 − 10·S`, **not** a percentage, so it is loaded unscaled; dividing it by 100 like a
volatility index would corrupt every skewness derived from it.

## Limitations

- **The strike dimension is calibrated to one number per day.** A quoted option chain gives dozens
  of points per maturity. Every fit statistic here should be read against that.
- **VIX1D is downloaded but excluded from the calibration backbone**, because it only begins in May
  2022 and including it would make the sample inconsistent across time.
- **A global SSVI is deliberately rigid.** Freeing more parameters per slice would fit better and
  say less about whether the family describes the market.
- The Cboe indexes are model-free variance-swap-style measures, not traded option prices. A
  violation in the published curve would not by itself be a tradeable arbitrage.
- The sufficient conditions are sufficient, not necessary, so the constrained fit is conservative:
  some surfaces it rejects would in fact have been valid.
- No transaction costs, no bid-ask, no execution. This study prices a surface; it does not trade it.

**Results are research findings, not investment advice.**

## Origin

This reimplements a study that previously existed only as a Word report and an Excel workbook, both
preserved in `deliverables/` (kept out of version control as large binaries). That original's
universe sheet listed a `base_iv`, `shock_beta` and `rho` per ticker — the parameters of a synthetic
surface generator — and its analysis code was never delivered.

This implementation replaces the generated surfaces with observed inputs: Cboe's published term
structure for the maturity dimension and its published skew index for the strike dimension. The
numbers therefore differ from the report's, because they now rest on market data.

## License

[MIT](LICENSE).
