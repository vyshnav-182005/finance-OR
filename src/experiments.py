"""Robustness / ablation experiments (CLAUDE.md §21, §28). Each is a separate named run
with ONE change against the primary configuration; none is used to pick the primary model.
Transaction-cost sensitivity needs no re-run: see metrics/cost_sensitivity.csv of each run.

Output: results/runs/<name>/... per experiment and results/experiments_summary.csv
(full-period OUT-OF-SAMPLE net metrics of the Proposed strategy + per-window Sharpe wins).
"""
import sys

import pandas as pd

from backtest import load_data, run, with_overrides
from evaluation import evaluate
from preprocessing import ROOT, load_config

W = lambda sh, re, li, vo, dd: {"score_weights": {"Sharpe": sh, "Return": re, "Liquidity": li,
                                                  "Volatility": vo, "MaxDrawdown": dd}}
EXPERIMENTS = {
    "primary": {},
    # K sensitivity: K fixed in every window instead of silhouette-chosen
    **{f"K{k}_fixed": {"k": k} for k in (5, 6, 7, 8)},
    # score-weight sensitivity (sum to 1)
    "weights_risk": {"overrides": {"selection": W(0.25, 0.10, 0.10, 0.30, 0.25)}},
    "weights_return": {"overrides": {"selection": W(0.25, 0.40, 0.10, 0.10, 0.15)}},
    "weights_equal": {"overrides": {"selection": W(0.2, 0.2, 0.2, 0.2, 0.2)}},
    # feature sensitivity: Model B = six features + 12-month momentum in K-Means
    "momentum_features": {"momentum": True},
    # constraint sensitivity
    "max2_per_cluster": {"overrides": {"selection": {"max_per_cluster": 2}}},
    "maxw_10pct": {"overrides": {"goal_programming": {"max_weight": 0.10}}},
    "maxw_25pct_minw_0": {"overrides": {"goal_programming": {"max_weight": 0.25, "min_weight": 0.0}}},
}


def summarise(name, res, primary_holdings):
    fp = res["full_period"].set_index("Strategy")
    p = fp.loc["Proposed"]
    cmp = res["comparisons"].query("Metric == 'Sharpe'").set_index("Baseline")
    row = {"Experiment": name, **{k: p[k] for k in ["AnnReturn", "AnnVol", "Sharpe", "Sortino", "MaxDrawdown",
                                                    "Beta", "AvgTurnover"]},
           **{f"SharpeWinsVs_{b}": f"{cmp.ProposedWins[b]}/{cmp.Periods[b]}"
              for b in ["EW_Same15", "Random15", "EW_Universe", "NIFTY50"]}}
    hold = res["holdings"]
    if primary_holdings is not None:      # avg share of holdings shared with the primary model
        row["OverlapWithPrimary"] = (hold.groupby("WindowID").Stock.apply(set).combine(
            primary_holdings, lambda a, b: len(a & b) / len(a))).mean()
    return row


def main(names):
    base = load_config()
    data = load_data(base)
    rows, prim = [], None
    for name in names:
        e = EXPERIMENTS[name]
        cfg = with_overrides(base, e.get("overrides"))
        out = run(cfg, data, name=name, k=e.get("k"), momentum=e.get("momentum", False))
        res = evaluate(name, cfg, make_plots=(name == "primary"))
        port = out["portfolios"]
        res["holdings"] = port[port.Strategy == "Proposed"]
        if name == "primary":
            prim = res["holdings"].groupby("WindowID").Stock.apply(set)
        rows.append(summarise(name, res, prim))
    s = pd.DataFrame(rows)
    s.to_csv(ROOT / "results/experiments_summary.csv", index=False, float_format="%.4f")
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or list(EXPERIMENTS))
