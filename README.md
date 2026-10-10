# Finance-OR: K-Means + Integer Programming + Goal Programming on NSE

Walk-forward construction of a 15-stock NSE portfolio: behavioural K-Means clustering ->
cluster-diversified Integer Programming selection -> Goal Programming weights -> 1-year
out-of-sample buy-and-hold test. 22 windows: train 1996-1999 / test 2000, ... train
2017-2020 / test 2021 (Jan-Jun 2021; the data ends 2021-07-01).

## Run

Python: `C:\Users\vyshn\miniconda3\python` (pandas, numpy, scipy, scikit-learn, matplotlib, pyarrow).
No other solver is needed: the IP uses `scipy.optimize.milp` (HiGHS), the GP uses SLSQP.

```
python src/preprocessing.py   # data/processed/: cleaned panel, corporate actions, report
python src/features.py        # results/features/: universe + 6 features + Z per window
python src/clustering.py      # results/clustering/: K diagnostics, labels, cluster profiles
python src/experiments.py     # primary + all robustness runs -> results/runs/<name>/, summary
python src/dashboard.py       # results/dashboard/index.html: interactive results dashboard (opens in browser)
python tests/test_pipeline.py # unit checks (plus self-checks in integer_selection.py, goal_programming.py)
```
`python src/backtest.py` / `python src/evaluation.py` run the primary model alone.
Every parameter is in `configs/parameters.yaml`; a changed parameter is a new experiment.

## Method (all per window, training data only)

| Stage | Definition |
|---|---|
| Universe | traded on >= 80% of NIFTY days in the window, traded in its last 21 days, last close >= INR 5. No requirement to survive into the test year. |
| Features | Return = CAGR; Volatility = std * sqrt(252) (consecutive-day returns); Sharpe = (Return - 7%) / Vol; Beta vs NIFTY 50; MaxDrawdown as a positive loss; Liquidity = log(1 + mean daily traded value). Z-scored with this window's mean/std. |
| K-Means | K in 3..8 evaluated (silhouette, inertia, sizes). K chosen = max silhouette among K that keep the IP feasible (K >= 5 and enough members to take <= 3 per cluster). n_init 20, seed 42. |
| IP | max sum Score_i x_i; sum x = 15; <= 3 per cluster; >= 5 clusters. Score = 0.30 Z(Sharpe) + 0.20 Z(Return) + 0.10 Z(Liquidity) - 0.20 Z(Vol) - 0.20 Z(MaxDD). |
| GP | weighted GP, equal priorities, deviations normalised by target: Return >= 20%, Vol <= 20% (training covariance), Sharpe >= 0.9, Beta <= 1. 2% <= w_i <= 15%. Tie-break: lowest volatility. |
| Test | weights frozen at the last training close, buy-and-hold (drift) for the test year. Turnover = sum abs(w_new - w_drifted); cost 0.25% x turnover on the first test day. |

Baselines: EW same 15, Markowitz max-Sharpe same 15 (same bounds), Top-Sharpe 15, Top-Return 15,
Random 15 (+100 random draws per window), EW whole eligible universe, NIFTY 50, NIFTY 500.

## Primary results (out-of-sample, Jan 2000 - Jun 2021, net of 0.25% costs)

| Strategy | Ann. return | Vol | Sharpe | Sortino | Max DD | Beta | Turnover/yr |
|---|---|---|---|---|---|---|---|
| **Proposed (KM+IP+GP)** | **19.3%** | **20.3%** | **0.61** | **0.84** | 60.7% | 0.67 | 1.57 |
| EW same 15 | 18.2% | 22.9% | 0.49 | 0.67 | 72.1% | 0.80 | 1.58 |
| Markowitz same 15 | 19.4% | 23.6% | 0.53 | 0.72 | 66.8% | 0.77 | 1.51 |
| Top-Sharpe 15 | 8.7% | 26.0% | 0.07 | 0.09 | 83.8% | 0.88 | 1.31 |
| Top-Return 15 | 6.8% | 28.4% | -0.01 | -0.01 | 85.6% | 0.93 | 1.34 |
| Random 15 | 19.2% | 25.3% | 0.48 | 0.67 | 73.8% | 0.74 | 1.90 |
| EW universe | 18.5% | 21.9% | 0.53 | 0.70 | 70.2% | 0.75 | 0.48 |
| NIFTY 50 | 11.8% | 23.0% | 0.21 | 0.29 | 59.9% | 1.00 | - |
| NIFTY 500 | 12.0% | 23.0% | 0.22 | 0.30 | 68.0% | 0.97 | - |

Per-window paired tests (22 test years, `results/runs/primary/metrics/comparisons.csv`):

* **vs NIFTY 50 / 500:** higher Sharpe on average (+0.77 / +0.62; bootstrap 95% CI excludes 0,
  Wilcoxon p = 0.046 / 0.059). Wins in 12 / 14 of 22 years.
* **Drawdown:** smaller test-year max drawdown than every stock-based baseline in most years
  (21/22 vs EW same 15, 21/22 vs Random 15, 22/22 vs EW universe; all p < 0.01).
* **Return / Sharpe vs EW same 15, Random 15, EW universe, Markowitz:** differences are **not
  statistically significant** (p > 0.3, CIs span 0; ~50-60% win rates). The full-period
  Sharpe is the highest of all strategies, but most of it comes from lower risk, not higher return.
* Against 100 random 15-stock portfolios per year the proposed portfolio sits at the 54th
  (return) / 57th (Sharpe) percentile on average.
* Regimes (ex-post labels, analysis only): it loses less in Bear years (-19.9% vs -22.6% NIFTY 50
  and -24.2% EW same 15). In Bull years it trails the equal-weight portfolios on return but
  matches them on Sharpe.
* Costs: Sharpe 0.63 / 0.61 / 0.58 / 0.53 at 0 / 0.25 / 0.5 / 1% per unit of turnover. The ranking
  against the baselines does not change.

**Answer to the research question:** the framework delivers clearly better risk-adjusted
performance than the market indices and consistently lower drawdowns than simple stock-picking
rules. Its edge over naive equal-weight diversification of small/mid caps is in risk, not in
statistically significant extra return. Equal-weight NSE stocks themselves beat the
cap-weighted indices by ~7%/yr over 2000-2021. Naive "past winner" rules (Top-Sharpe / Top-Return)
fail badly out of sample.

## Robustness (`results/experiments_summary.csv`, one change per run)

| Experiment | Sharpe | Ann. return | Max DD | Holdings shared with primary |
|---|---|---|---|---|
| primary (silhouette K) | 0.61 | 19.3% | 60.7% | 100% |
| K fixed 5 / 6 / 7 / 8 | 0.45 / 0.53 / 0.50 / 0.45 | 16-18% | 56-66% | 49-76% |
| weights risk-focused | 0.69 | 20.2% | 59.9% | 64% |
| weights return-focused | 0.43 | 16.4% | 69.1% | 61% |
| weights equal (0.2 each) | 0.62 | 19.2% | 60.9% | 69% |
| + momentum in K-Means | 0.52 | 17.6% | 68.2% | 45% |
| max 2 per cluster | 0.49 | 17.8% | 57.6% | 41% |
| max weight 10% | 0.62 | 20.1% | 65.8% | 100% |
| max weight 25%, min 0% | 0.61 | 19.1% | 57.0% | 100% |

Every variant keeps Sharpe 0.43-0.69, well above NIFTY 50 (0.21). Risk-focused score weights
help and return-focused weights hurt, which is consistent with Top-Return failing. These are
sensitivity results only. The primary configuration was fixed before testing and is not
re-chosen from this table.

## Assumptions and limitations

* **Corporate actions:** raw prices are unadjusted. Splits and bonuses are detected heuristically
  (standard ratios over <= 5 days) and neutralised. Rights issues and demergers are not
  adjusted, and a genuine crash at a standard ratio would be misread.
* **No dividends:** stock returns and both NIFTY indices are price returns.
* **Survivorship:** the universe is built only from data available at formation, and test-year
  survival is never required. But the dataset has **no delisting returns**: a holding that stops
  trading is held at its last price as cash (count recorded in `windows.csv`). Also the dataset
  covers ~1,700 scrips that may under-represent firms that died early, so some survivorship bias
  can remain.
* **Risk-free rate:** constant 7% (no T-bill series in the data).
* **Data start:** stock files start in 1996, so the first training window is 1996-1999, not 1990.
* **Weak clusters:** silhouette is only 0.21-0.28, and K = 3 often scores best but cannot satisfy
  ">= 5 clusters". With 15 stocks and <= 3 per cluster, the ">= 5 clusters" constraint is always implied.
* **GP return goal** uses the weighted training CAGR (in-sample, optimistic: the goals are
  met in training but the realised test returns are much lower).
* The volatility goal (<= 20%) is missed in 9/22 windows, mostly 2000-2003. The other goals are always met.
* "Year-wise/recency weighting" from the proposal slides is not implemented (not in the spec).
* 2021 is a half-year test period; per-window metrics are annualised.
