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
from integer_selection import v2_rules
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
    # EXTENSION v2 (selection_v2): slack-adjusted, cluster-quality-aware selection. Motivated by
    # the zero-slack constraint flaw, not by results; every N is reported. In v2 runs the
    # TopSharpe15/TopReturn15/Random15 baselines hold N stocks (same N as the proposed model).
    **{f"v2_N{n}": {"v2": {"n_stocks": n}} for n in (15, 10, 12, 20)},
    "v2_N15_slack_only": {"v2": {"exclude_negative_sharpe_clusters": False}},
    # v3 = v2 + exclusion-aware K choice (fixes v2's K=4 interaction, where the exclusion could
    # never apply and every cluster was again forced to supply >= 3 stocks). Rule fixed a priori.
    # A v3 slack-only run would equal v2_N15_slack_only (no exclusion -> nothing to be aware of).
    **{f"v3_N{n}": {"v2": {"n_stocks": n, "exclusion_aware_k": True}} for n in (15, 10, 12, 20)},
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
    prim_file = ROOT / "results/runs/primary/portfolios.csv"
    if "primary" not in names and prim_file.exists():
        p = pd.read_csv(prim_file)
        prim = p[p.Strategy == "Proposed"].groupby("WindowID").Stock.apply(set)
    for name in names:
        e = EXPERIMENTS[name]
        cfg = with_overrides(base, e.get("overrides"))
        if "v2" in e:
            cfg["selection"] = v2_rules({**base["selection_v2"], **e["v2"]}, cfg["selection"])
        out = run(cfg, data, name=name, k=e.get("k"), momentum=e.get("momentum", False))
        res = evaluate(name, cfg, make_plots=(name == "primary"))
        port = out["portfolios"]
        res["holdings"] = port[port.Strategy == "Proposed"]
        if name == "primary":
            prim = res["holdings"].groupby("WindowID").Stock.apply(set)
        rows.append(summarise(name, res, prim))
    s = pd.DataFrame(rows)
    out = ROOT / "results/experiments_summary.csv"
    if out.exists():                       # keep rows of experiments not re-run now
        old = pd.read_csv(out)
        s = pd.concat([old[~old.Experiment.isin(s.Experiment)], s], ignore_index=True)
    s.to_csv(out, index=False, float_format="%.4f")
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or list(EXPERIMENTS))
