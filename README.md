# Finance-OR: K-Means + Integer Programming + Goal Programming on NSE

Builds a 15-stock NSE portfolio each year. Training uses only the previous 4 years and testing uses the next year (22 windows, test years 2000 to mid-2021):

1. **K-Means** groups stocks by 6 features: return, volatility, Sharpe, beta, max drawdown and liquidity.
2. **Integer Programming** picks 15 stocks, at most 3 per cluster.
3. **Goal Programming** sets the weights, each between 2% and 15%.
4. The portfolio is held for the next year, out of sample, with 0.25% transaction costs.

## Run

Use `C:\Users\vyshn\miniconda3\python`. All parameters are in `configs/parameters.yaml`.

```
python src/preprocessing.py
python src/features.py
python src/clustering.py
python src/experiments.py      # primary model + robustness runs -> results/runs/
python src/dashboard.py        # interactive dashboard -> results/dashboard/index.html
python tests/test_pipeline.py
```

## Results (out-of-sample, 2000 to mid-2021)

| Strategy | Ann. return | Sharpe | Max DD |
|---|---|---|---|
| **Proposed** | **19.3%** | **0.61** | 60.7% |
| Equal-weight, same 15 stocks | 18.2% | 0.49 | 72.1% |
| Random 15 | 19.2% | 0.48 | 73.8% |
| NIFTY 50 | 11.8% | 0.21 | 59.9% |

- Beats NIFTY 50/500 on Sharpe, and the gap is statistically significant.
- Its drawdowns are lower than every other stock-picking baseline in almost every year.
- Its edge over equal-weight or random picks comes from lower risk. The difference in return is not significant.

## Limitations

- Prices are unadjusted. Splits and bonuses are detected heuristically, and dividends are not included.
- There are no delisting returns, so some survivorship bias may remain.
- The risk-free rate is a constant 7%.
- The stock data starts in 1996.
