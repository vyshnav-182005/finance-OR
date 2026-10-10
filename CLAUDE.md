# Finance-OR: Multi-Objective Stock Selection & Portfolio Construction

## 1. Project Overview

This project develops a systematic framework for stock selection and portfolio
construction for the Indian/NSE market by combining:

- Financial feature engineering
- K-Means behavioral clustering
- Integer Programming (IP)
- Goal Programming (GP)
- Walk-forward out-of-sample backtesting

The main objective is to construct a diversified portfolio of 15 stocks while
balancing return, risk, risk-adjusted performance, market sensitivity, and
liquidity.

The project is NOT a stock-price prediction model.

K-Means is used to group stocks according to financial behavior and support
diversification.

---

# 2. Dataset

Primary dataset:

- NSE historical stock data
- Daily OHLCV data
- Historical coverage: approximately 1990–2021
- NSE index data, including NIFTY 50
- NSE symbol/company mapping

NIFTY 50 is used for:

- Market-relative calculations
- Beta calculation
- Primary market benchmark
- Market/regime analysis where applicable

Do not assume the dataset contains fields that have not been verified.
Inspect the actual files/schema before writing preprocessing code.

---

# 3. Overall Pipeline

The intended pipeline is:

Raw NSE Data
    ↓
Data Preprocessing
    ↓
Investable Universe Construction
    ↓
Feature Engineering
    ↓
Z-score Standardization
    ↓
K-Means Behavioral Clustering
    ↓
Integer Programming
    ↓
Select exactly 15 stocks
    ↓
Goal Programming
    ↓
Determine portfolio weights
    ↓
1-year Out-of-Sample Testing
    ↓
Performance Evaluation
    ↓
Roll the window forward
    ↓
Repeat

---

# 4. Walk-Forward Backtesting

Use a 4-year rolling training window followed by a 1-year out-of-sample test.

Example:

Window 1:
1990–1993 → training/construction
1994 → test

Window 2:
1991–1994 → training/construction
1995 → test

Window 3:
1992–1995 → training/construction
1996 → test

Continue until the latest possible period.

IMPORTANT:

Every window must be independently re-estimated.

Do NOT fit any component using future/test data.

For a window:

Training data
    ↓
Preprocessing decisions
    ↓
Feature calculation
    ↓
Scaling
    ↓
K-Means
    ↓
Integer Programming
    ↓
Goal Programming
    ↓
Portfolio
    ↓
Test on next year only

The test year must remain completely unseen during portfolio construction.

---

# 5. Data Leakage Rules

Avoid look-ahead bias at every stage.

For each training window:

- Do not use future prices.
- Do not calculate feature statistics using the test period.
- Do not calculate scaling parameters using the test period.
- Do not train K-Means using test-period features.
- Do not use test-period returns when selecting stocks.
- Do not use test-period information when determining portfolio weights.
- Do not use future liquidity information.
- Do not use future corporate-action information unless it represents
  information that would actually have been known at the portfolio formation
  date.

Any transformation that learns parameters must be fitted only on the training
window.

---

# 6. Data Preprocessing

Before feature engineering:

1. Load the relevant NSE stock data.
2. Inspect the actual schema.
3. Standardize date formats.
4. Sort data chronologically.
5. Remove duplicate observations.
6. Identify invalid OHLCV observations.
7. Handle missing values appropriately.
8. Align stock and NIFTY 50 trading dates.
9. Handle corporate actions where supported by the dataset.
10. Construct an investable universe.

Do not blindly fill missing stock prices.

For price/return calculations, preserve the distinction between:
- Missing observations because the stock did not trade
- Missing observations because of data problems
- Dates before/after listing

---

# 7. Investable Universe

The stock universe should reflect what could realistically have been invested
in at the beginning/end of each training window.

Consider:

- Sufficient historical observations
- Listing availability
- Delisting availability
- Trading activity
- Liquidity

Avoid survivorship bias.

Do not automatically restrict historical analysis to companies that survived
until the end of the entire dataset.

If the dataset does not provide enough information to fully address
survivorship bias, explicitly document this limitation instead of pretending
it has been solved.

---

# 8. Feature Engineering

The core six financial features are:

1. Return
2. Volatility
3. Sharpe Ratio
4. Beta
5. Maximum Drawdown
6. Liquidity

These six features are the primary behavioral features used for K-Means.

Raw OHLCV is NOT directly used as the K-Means feature matrix.

The relationship is:

OHLCV
  ↓
Financial calculations
  ↓
Six engineered features
  ↓
K-Means

---

## 8.1 Daily Return

For stock i:

r(i,t) = P(i,t) / P(i,t-1) - 1

Use the appropriate price field after verifying whether the dataset has
corporate-action-adjusted prices.

The return metric used for stock comparison must be calculated consistently
across all stocks within a window.

---

## 8.2 Return

Calculate the historical return over the training window.

Possible definitions include cumulative return or annualized return.

Before implementation, inspect the existing project code/requirements and
maintain one consistent definition across all windows.

Do not mix different return definitions between selection and evaluation
without documenting it.

---

## 8.3 Volatility

Use standard deviation of daily returns.

Annualized volatility can be calculated as:

Volatility = std(daily_returns) × sqrt(252)

Higher volatility represents higher risk.

---

## 8.4 Sharpe Ratio

Sharpe Ratio represents risk-adjusted return.

Conceptually:

Sharpe = (Return - RiskFreeRate) / Volatility

The exact risk-free-rate assumption must be documented.

If a risk-free series is unavailable, do not silently invent a sophisticated
risk-free dataset. Use a clearly documented assumption.

Higher Sharpe is preferred.

---

## 8.5 Beta

Use NIFTY 50 as the market benchmark.

Beta_i =
Covariance(stock_i_returns, NIFTY_returns)
/
Variance(NIFTY_returns)

Beta represents sensitivity to overall market movements.

The same training window must be used for stock and NIFTY returns.

Do not calculate beta using future/test-period data.

---

## 8.6 Maximum Drawdown

For each stock:

RunningPeak_t = max(P_1, ..., P_t)

Drawdown_t = (P_t - RunningPeak_t) / RunningPeak_t

Maximum Drawdown = minimum(Drawdown_t)

Maximum drawdown represents the largest peak-to-trough decline.

Lower absolute downside risk is preferred.

Be consistent about whether drawdown is stored as a negative percentage or
as a positive loss magnitude.

---

## 8.7 Liquidity

Preferred liquidity measure:

Average Traded Value = mean(Price × Volume)

Average traded value is preferred over raw volume because it represents the
monetary value traded.

Liquidity may be highly skewed.

If appropriate, use:

log(1 + Average Traded Value)

before standardization.

Do not make this transformation without documenting it.

---

# 9. Feature Standardization

The six features have different units and scales.

Before K-Means, standardize them using Z-scores:

Z_ij = (X_ij - mean_j) / std_j

IMPORTANT:

The mean and standard deviation must be calculated separately within each
training window.

Never calculate global 1990–2021 scaling parameters and apply them to all
historical windows.

The K-Means feature matrix is:

Rows = stocks
Columns = six standardized financial features

Example:

Stock | Return | Volatility | Sharpe | Beta | Drawdown | Liquidity
A     | ...
B     | ...
C     | ...

---

# 10. K-Means Clustering

Purpose:

K-Means is used for behavioral diversification, NOT price prediction.

Stocks with similar financial characteristics should be grouped together.

Input:

- Standardized Return
- Standardized Volatility
- Standardized Sharpe
- Standardized Beta
- Standardized Maximum Drawdown
- Standardized Liquidity

Each stock receives a cluster label.

Example:

Stock A → Cluster 2
Stock B → Cluster 4
Stock C → Cluster 1

---

# 11. Number of Clusters

Do NOT blindly hard-code K = 5 without justification.

Prefer a data-driven approach.

Evaluate candidate values such as:

K = 3, 4, 5, 6, 7, 8

Use clustering diagnostics such as:

- Silhouette score
- Elbow/inertia
- Cluster-size balance
- Economic interpretability

The selected K should be determined using only the current training window.

If the project ultimately fixes K = 5 for methodological simplicity, document
why and perform sensitivity analysis with alternative K values.

Do not choose K based on test-year performance.

---

# 12. Cluster Interpretation

K-Means cluster numbers are arbitrary.

Cluster 1 does NOT inherently mean "low risk".
Cluster 2 does NOT inherently mean "high return".

After clustering, calculate cluster-level feature averages and interpret the
behavior of each cluster.

For example:

Cluster X:
- Higher return
- Higher volatility
- Higher beta

Cluster Y:
- Lower volatility
- Lower drawdown
- Moderate return

Use this interpretation only after fitting K-Means.

---

# 13. Integer Programming

Purpose:

Integer Programming determines WHICH stocks to select.

Binary decision variable:

x_i = 1 → stock i selected
x_i = 0 → stock i not selected

The portfolio must select exactly 15 stocks:

sum(x_i) = 15

Diversification constraints:

- Maximum 3 stocks per cluster
- At least 5 clusters represented

The selected stocks should be based on the selection score and constraints.

---

# 14. Selection Score

The current conceptual score is:

Score_i =
w1 * Z(Sharpe_i)
+ w2 * Z(Return_i)
+ w3 * Z(Liquidity_i)
- w4 * Z(Volatility_i)
- w5 * Z(Drawdown_i)

The exact weights must be explicitly documented.

Weights should satisfy:

sum(weights) = 1

Do not arbitrarily change weights based on test-year results.

Preferred analysis:

- Base/balanced weights
- Risk-focused weights
- Return-focused weights
- Equal-weight sensitivity case

Evaluate whether conclusions remain stable.

---

# 15. Goal Programming

Purpose:

Goal Programming determines HOW MUCH to invest in each selected stock.

Integer Programming:
    Which stocks?

Goal Programming:
    What weights?

Basic portfolio constraint:

sum(w_i) = 1

For unselected stocks:

w_i = 0

Possible portfolio-level objectives/targets:

- Target Return
- Maximum Volatility
- Minimum Sharpe Ratio
- Maximum Beta

Portfolio risk should account for covariance between selected stocks.

Covariance must be estimated using training-period returns only.

---

# 16. Portfolio Weight Constraints

Consider practical constraints such as:

- Maximum individual stock weight
- Minimum weight for selected stocks
- Full investment constraint
- Optional turnover constraint

Do not add arbitrary constraints without documenting their purpose.

The goal is to prevent unrealistic concentration.

---

# 17. Transaction Costs and Turnover

Turnover should be evaluated between consecutive portfolio rebalances.

A basic turnover measure:

Turnover_t =
sum_i |w_i,t - w_i,t-1|

If transaction costs are modeled:

TransactionCost_t =
cost_rate × turnover_t

Net return should account for transaction costs when performing the
realistic backtest.

The transaction-cost assumption must be documented and sensitivity-tested.

---

# 18. Backtesting

The primary test is one-year out-of-sample performance.

For each window:

1. Construct portfolio using training data.
2. Freeze portfolio at the formation point.
3. Apply it to the next year's unseen data.
4. Calculate realized performance.
5. Move the training window forward.
6. Recalculate everything.
7. Repeat.

Do not re-fit the model using information from the test year unless the
method explicitly defines an allowed rebalance point.

---

# 19. Baseline Strategies

Compare the proposed strategy against:

1. Proposed:
   K-Means + Integer Programming + Goal Programming

2. Equal-weight same 15 stocks

3. Top-Sharpe 15

4. Top-Return 15

5. Random 15

6. NIFTY 50

7. NIFTY 500

8. Markowitz/classical portfolio benchmark where implemented

Consider an equal-weight broad-universe portfolio as an additional baseline
if the data and implementation permit it.

The purpose of baselines is to determine whether the proposed framework adds
value over simpler approaches.

---

# 20. Evaluation Metrics

Primary evaluation should focus on out-of-sample risk-adjusted performance.

Metrics:

- Cumulative Return
- Annualized Return
- Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Beta
- Turnover
- Transaction-cost-adjusted return

Do not judge the model only by cumulative return.

The project's primary goal is consistent risk-adjusted performance.

---

# 21. Robustness Analysis

Where computationally feasible, perform:

### K sensitivity
Compare different K values.

### Feature sensitivity
Compare:

Model A:
Six core features

Model B:
Six features + Momentum

Do not add momentum to the primary model unless it is clearly documented
as an extension.

### Score-weight sensitivity
Test multiple reasonable weight configurations.

### Portfolio constraint sensitivity
Test reasonable stock/cluster/weight constraints.

### Transaction-cost sensitivity
Test multiple realistic transaction-cost assumptions.

---

# 22. Market-Regime Analysis

Use NIFTY 50 data to investigate whether performance changes across market
conditions.

Possible regimes:

- Bull
- Bear
- High volatility
- Low volatility

This is an analysis/extension, not a reason to introduce future information
into portfolio construction.

Regime labels for an out-of-sample period must not use future information
when those labels are intended to influence portfolio construction.

---

# 23. Statistical Evaluation

Do not rely only on one cumulative return number.

Report:

- Number of test periods
- Number of periods outperforming each baseline
- Average annualized performance
- Distribution of annual returns
- Risk-adjusted metrics

Where appropriate, use statistical tests or bootstrap confidence intervals
to assess whether performance differences are meaningful.

---

# 24. Coding Principles

Use Python.

Preferred libraries:

- pandas
- numpy
- scikit-learn
- scipy
- Pyomo / PuLP
- Gurobi where available
- CVXPY
- matplotlib

Keep the implementation modular.

Suggested structure:

data/
    raw/
    processed/

src/
    preprocessing/
    features/
    clustering/
    optimization/
    backtesting/
    evaluation/

configs/
    parameters.yaml

results/
    windows/
    portfolios/
    metrics/
    plots/

CLAUDE.md

---

# 25. Recommended Modules

Suggested modules:

src/preprocessing.py
src/features.py
src/clustering.py
src/integer_selection.py
src/goal_programming.py
src/backtest.py
src/evaluation.py
src/utils.py

Keep functions small and testable.

Do not put the entire pipeline in one notebook/script.

---

# 26. Reproducibility

Set random seeds for K-Means and random baselines.

Record for every window:

- Training period
- Test period
- Number of eligible stocks
- Feature definitions
- Scaling parameters
- Selected K
- Cluster sizes
- Selected stocks
- Selection scores
- Portfolio weights
- Portfolio return
- Risk metrics
- Turnover
- Transaction costs
- Solver status

Save these outputs so results can be reproduced.

---

# 27. Important Implementation Rule

Never silently change the methodology because a result looks bad.

If an experiment changes:

- Feature definition
- K
- Score weights
- Portfolio constraints
- Transaction costs
- Optimization objectives

record it as a separate experiment.

Do not tune hyperparameters against the final test period.

---

# 28. Experimental Design

Distinguish clearly between:

PRIMARY MODEL
K-Means + IP + GP

BASELINES
Simple portfolio strategies and market indices

ABLATION/ROBUSTNESS EXPERIMENTS
- Different K
- Different score weights
- Different features
- Different constraints
- Transaction-cost assumptions

Do not confuse experimental improvements with the original proposed model.

---

# 29. Current Project Research Question

Core question:

Can a combination of behavioral K-Means clustering, Integer Programming,
and Goal Programming construct a diversified portfolio of 15 NSE stocks
that delivers consistent risk-adjusted out-of-sample performance compared
with simpler stock-selection strategies and market benchmarks?

---

# 30. How Claude Code Should Work on This Project

Before modifying code:

1. Inspect the existing repository.
2. Inspect the dataset schema.
3. Understand existing implementation.
4. Do not overwrite working code without checking dependencies.
5. Explain major methodological changes before implementing them.

When implementing:

- Prefer reproducible code.
- Add comments for important financial/optimization logic.
- Validate intermediate outputs.
- Check for data leakage.
- Check dimensions and missing values.
- Check optimization feasibility.
- Save intermediate results.
- Use deterministic random seeds where appropriate.

When debugging:

- First identify the root cause.
- Do not hide errors with arbitrary exception handling.
- Do not silently drop data unless justified.
- Report how many stocks/rows were removed and why.

When producing results:

- Clearly distinguish training-period calculations from test-period results.
- Never report an in-sample metric as an out-of-sample result.
- Preserve the exact window dates.

---

# 31. Key Conceptual Distinction

Always remember:

K-Means:
    "Which stocks behave similarly?"

Integer Programming:
    "Which 15 stocks should be selected?"

Goal Programming:
    "How should capital be allocated among those 15?"

Backtesting:
    "Does the resulting strategy work on unseen future data?"

These are separate stages and should remain conceptually separate in both
the code and the explanation.

---

# 32. Current Preferred End-to-End Workflow

For each rolling window:

1. Identify the training period.
2. Identify the following one-year test period.
3. Build the historical investable universe.
4. Clean and align stock/NIFTY data.
5. Calculate daily returns.
6. Calculate the six financial features.
7. Remove invalid/insufficient observations.
8. Standardize features using training-window statistics.
9. Determine K using training data only.
10. Fit K-Means.
11. Assign stocks to behavioral clusters.
12. Calculate stock selection scores.
13. Run Integer Programming.
14. Select exactly 15 stocks.
15. Apply cluster diversification constraints.
16. Run Goal Programming.
17. Determine portfolio weights.
18. Freeze the portfolio.
19. Evaluate on the next one-year test period.
20. Calculate return, volatility, Sharpe, Sortino, drawdown, beta and turnover.
21. Apply transaction-cost assumptions where required.
22. Store all results.
23. Roll the window forward by one year.
24. Repeat.

---

# 33. Golden Rule

The most important rule in this project:

NO FUTURE INFORMATION.

For any test year Y:

All decisions affecting the portfolio for year Y must be based only on
information available before year Y.

This includes:

- Data preprocessing
- Feature engineering
- Scaling
- K-Means
- K selection
- Selection-score weights
- Integer Programming
- Goal Programming
- Portfolio weights

The test year exists only to evaluate the already-constructed portfolio.