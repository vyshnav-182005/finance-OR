# Selection v2 / v3: slack-adjusted, cluster-quality-aware IP (extensions)

**Status:** robustness/extension experiments. **The primary model (`selection` block, `results/runs/primary`)
is unchanged and remains the reported result.** Its files were hash-checked before and after these runs.
v2 and v3 were motivated by a constraint-design flaw that we found by inspecting the constraints. They
were not motivated by test-year performance, and every rule was fixed before its runs.

## 1. The flaw

The primary IP requires `N = 15`, `max_per_cluster = 3` and `min_clusters = 5`. When K = 5 (chosen in
13/22 windows), 5 × 3 = 15 leaves **zero slack**: the IP must take exactly 3 stocks from every cluster,
however poor that cluster is. In window 21 (train 2016–2019, test 2020) it was forced to take
HINDCOPPER, RIIL and SRIPIPES from cluster 0 (mean return −26%, mean Sharpe −0.65). In that case the
selection score has no influence on which clusters are used.

## 2. The v2 fixes (`selection_v2` in `configs/parameters.yaml`)

1. **Slack-adjusted rules.** `max_per_cluster = ceil(N/4)` and `min_clusters = 4`. For N = 15 this is
   ≤ 4 per cluster.
2. **Cluster-quality filter.** Using **training-window features only**, compute each cluster's mean raw
   Sharpe after K-Means. Clusters with a mean below 0 are excluded: all their stocks get `x_i = 0`
   (upper bound 0 in the MILP).

**Fallback.** If fewer than `min_eligible_clusters = 4` clusters remain, or the IP becomes infeasible,
the exclusion is dropped for that window and logged. That window then uses only the slack-adjusted rules.

In v2, K is still chosen by maximum silhouette among K values that are IP-feasible under the slack
rules, using training data only.

Goal Programming weight limits (2%–15%) are feasible for every N tested: 0.02 × 20 = 0.4 ≤ 1 and
0.15 × 10 = 1.5 ≥ 1. No scaling was needed. The code would widen the limits and record it if a
future N required that.

## 3. v2 per window (identical for every N)

| | Windows (of 22) |
|---|---|
| ≥ 1 cluster with negative mean Sharpe | 19 |
| Exclusion applied | **3** (2003, 2004, 2018) |
| Fallback triggered | **16** |
| No negative-Sharpe cluster | 3 (2006–2008) |
| K chosen = 4 / 5 / 6 / 7 / 8 | 13 / 4 / 3 / 1 / 1 |

**Interaction (reported, not tuned):** `min_clusters = 4` makes K = 4 feasible, and the silhouette
score picks K = 4 in 13/22 windows. That causes two problems:
- **The forcing comes back.** With K = 4 and ≤ 4 per cluster there are 16 slots for 15 stocks
  (slack 1), so the IP must still take at least 3 stocks from every cluster, including poor ones.
- **The filter cannot act.** Excluding any cluster leaves 3, below `min_eligible_clusters = 4`, so the
  exclusion always falls back.

In 2020 (K = 4, clusters 0 and 1 negative) negative cluster 1 again supplied 3 stocks (HINDCOPPER,
RIIL, SUNDRMBRAK), though now at the 2% weight floor. **v2 therefore does not fix the original flaw.**
This led to v3 (section 4).

## 4. v3: exclusion-aware choice of K

**Rule (fixed before running):** `exclusion_aware_k: true`. K = max silhouette among the candidate K
where the exclusion is feasible: at least `min_eligible_clusters` clusters with mean training Sharpe ≥ 0,
and enough capacity to hold N stocks under `max_per_cluster`. If no candidate K qualifies, use v2's
choice (slack-only feasible K) and fall back. Everything else is identical to v2. A v3 slack-only run
would equal `v2_N15_slack_only`, because with no exclusion there is nothing for K to be aware of.

| Per window (N = 15) | v2 | v3 |
|---|---|---|
| Exclusion applied | 3 | **11** |
| Fallback | 16 | **8** (2000, 2001, 2002, 2009, 2012, 2014, 2020, 2021) |
| No negative cluster | 3 | 3 |
| K = 4 chosen | 13 | 7 |
| Selected stocks with negative training Sharpe (all windows) | 32 / 330 | 28 / 330 (primary: 38 / 330) |

The pattern across N is the same (N = 10: 12 applied, 7 fallback).

**Why 8 windows still fall back (data, not code):** in those training windows 64–87% of eligible stocks
have a Sharpe ratio below zero against the 7% risk-free rate. For every K from 3 to 8 only 1–3 clusters
have a non-negative mean Sharpe (2020: 2 clusters at every K; 2012: 1 at every K). Requiring
≥ 4 clusters and excluding every negative-Sharpe cluster are then mutually exclusive, and no K changes
that. **2020 is one of these windows, so its portfolio is unchanged from v2.**

## 5. Results: full-period out-of-sample, net of 0.25% costs, Proposed strategy

| Run | Sharpe | Ann. return | Ann. vol | Max DD | Sharpe wins vs EW_Same | vs Random | vs EW_Universe | vs NIFTY50 |
|---|---|---|---|---|---|---|---|---|
| **primary (reported)** | **0.606** | **19.3%** | **20.3%** | **60.7%** | 12/22 | 13/22 | 12/22 | 12/22 |
| v2_N15 | 0.543 | 18.0% | 20.2% | 58.1% | 11/22 | 13/22 | 12/22 | 14/22 |
| v2_N15_slack_only | 0.562 | 18.2% | 20.0% | 58.1% | 10/22 | 13/22 | 12/22 | 14/22 |
| v2_N10 | 0.686 | 22.0% | 21.9% | 56.7% | 11/22 | 14/22 | 11/22 | 16/22 |
| v2_N12 | 0.636 | 20.6% | 21.4% | 54.3% | 12/22 | 15/22 | 12/22 | 15/22 |
| v2_N20 | 0.708 | 20.5% | 19.1% | 56.2% | 14/22 | 16/22 | 12/22 | 14/22 |
| v3_N15 | 0.548 | 17.9% | 19.9% | 58.1% | 10/22 | 11/22 | 12/22 | 15/22 |
| v3_N10 | 0.673 | 21.6% | 21.6% | 56.7% | 11/22 | 14/22 | 11/22 | 18/22 |
| v3_N12 | 0.698 | 21.7% | 21.0% | 54.3% | 11/22 | 14/22 | 12/22 | 16/22 |
| v3_N20 | 0.676 | 19.7% | 18.8% | 56.2% | 11/22 | 16/22 | 11/22 | 15/22 |

Full rows: `results/experiments_summary.csv`. Per-run detail: `results/runs/v2_*/` and `v3_*/`. The
windows.csv files record N, MaxPerCluster, MinClusters, NegSharpeClusters, ExclusionApplied,
ExclusionFallback, KExclusionFeasible (v3), IP_Status and GP_Status for every window.

Notes for reading the table:
- In v2/v3 runs, the EW_Same, TopSharpe, TopReturn and Random baselines hold **N** stocks. The column
  names still say "15". EW_Universe and NIFTY50 are the same in every run.
- At N = 15, neither v2 (0.543) nor v3 (0.548) beats the primary model's Sharpe (0.606). Max drawdown
  is slightly lower and wins vs NIFTY50 are higher.
- Across N, Sharpe ranges from 0.54 to 0.71 and is not monotonic in N. These are single 22-period
  paths with no significance test. **All N values are reported. No N is selected from these results.**

One GP solve in each of v2_N12, v2_N20, v3_N12 and v3_N20 ended in SLSQP mode 8 ("positive directional
derivative") at points that meet every constraint to about 1e-9. `goal_programming.py` now accepts
mode 8 only when that feasibility check passes. The primary run never hits this case.

## 6. Conclusion

- **v2** removes the zero slack at K ≥ 5. However, its K = 4 choice brings the forcing back and stops
  the filter acting, so on its own v2 does not fix the flaw.
- **v3** fixes the K-selection side: the exclusion now applies whenever *some* K allows it (11/22
  windows, vs 3).
- In the remaining 8 windows the market itself leaves too few non-negative clusters. There, the
  "≥ 4 clusters" diversification rule and the "negative Sharpe = excluded" quality rule genuinely
  conflict, and the fallback keeps diversification. Resolving this needs a different, a-priori rule
  (for example, exclude negative clusters worst-first only until `min_eligible_clusters` remain).
  That would be a separate labelled experiment.
- **All N values are reported and none is selected from these results. The primary model remains the
  original reported result.**
