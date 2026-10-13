"""Integer Programming: WHICH stocks (exactly n_stocks, cluster-diversified).

    max  sum_i Score_i x_i
    s.t. sum_i x_i                  = n_stocks
         sum_{i in k} x_i          <= max_per_cluster      for every cluster k
         y_k                       <= sum_{i in k} x_i     (k counts only if a stock is picked)
         sum_k y_k                 >= min_clusters
         x_i, y_k in {0, 1}

Score_i = sum_f w_f * sign_f * Z_f(i) with sign -1 for Volatility and MaxDrawdown (both
stored so higher = worse). Z-scores come from the training window only.
Solved with scipy.optimize.milp (HiGHS branch-and-bound).

v2 extension (selection_v2 in parameters.yaml, NOT the primary model): max_per_cluster may be
"auto" = ceil(N/4), and clusters whose mean raw training-window Sharpe is < 0 can be excluded
(x_i = 0 for their stocks). If too few clusters stay eligible or the IP is infeasible, the
exclusion is dropped for that window (fallback) and only the slack-adjusted rules apply.
"""
import math

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

SIGN = {"Sharpe": 1, "Return": 1, "Liquidity": 1, "Beta": 1, "Volatility": -1, "MaxDrawdown": -1}


def selection_score(feat, weights):
    assert np.isclose(sum(weights.values()), 1), "score weights must sum to 1"
    return sum(w * SIGN[f] * feat["Z_" + f] for f, w in weights.items())


def v2_rules(v2, base):
    """Effective selection dict for a v2 run: base `selection` (score weights) + v2 rules."""
    s = {**base, **v2}
    if s["max_per_cluster"] == "auto":
        s["max_per_cluster"] = math.ceil(s["n_stocks"] / 4)
    return s


def select_stocks(score, clusters, s, allowed=None):
    """score, clusters: aligned arrays; allowed: optional boolean mask (False -> x_i = 0).
    Returns (boolean mask of selected, solver status)."""
    score, clusters = np.asarray(score, float), np.asarray(clusters)
    n, ks = len(score), np.unique(clusters)
    K = len(ks)
    member = (clusters[None, :] == ks[:, None]).astype(float)          # K x n
    A = np.block([
        [np.ones((1, n)), np.zeros((1, K))],                            # sum x = n_stocks
        [member, np.zeros((K, K))],                                     # per-cluster cap
        [-member, np.eye(K)],                                           # y_k - sum_k x <= 0
        [np.zeros((1, n)), np.ones((1, K))],                            # sum y >= min_clusters
    ])
    lo = np.r_[s["n_stocks"], np.full(K, -np.inf), np.full(K, -np.inf), s["min_clusters"]]
    hi = np.r_[s["n_stocks"], np.full(K, s["max_per_cluster"]), np.zeros(K), np.inf]
    ub = np.r_[np.ones(n) if allowed is None else np.asarray(allowed, float), np.ones(K)]
    res = milp(c=-np.r_[score, np.zeros(K)], constraints=LinearConstraint(A, lo, hi),
               integrality=np.ones(n + K), bounds=Bounds(0, ub))
    if res.status != 0:
        raise RuntimeError(f"IP failed: {res.message}")
    return res.x[:n] > 0.5, res.message


def select_with_quality(score, clusters, sharpe, s):
    """IP with the optional v2 cluster-quality filter. sharpe = raw training-window Sharpe per
    stock (aligned). Returns (mask, solver status, info). Without
    s['exclude_negative_sharpe_clusters'] this is exactly select_stocks (primary model)."""
    clusters = np.asarray(clusters)
    info = {}
    if s.get("exclude_negative_sharpe_clusters"):
        means = pd.Series(np.asarray(sharpe, float)).groupby(clusters).mean()
        bad = means.index[means < 0]
        info = {"NegSharpeClusters": "|".join(map(str, bad)), "ExclusionApplied": False,
                "ExclusionFallback": False}
        if len(bad):
            if len(means) - len(bad) >= s["min_eligible_clusters"]:
                try:
                    sel, status = select_stocks(score, clusters, s, allowed=~np.isin(clusters, bad))
                    info["ExclusionApplied"] = True
                    return sel, status, info
                except RuntimeError as e:
                    print(f"  exclusion of clusters {list(bad)} infeasible ({e}); fallback")
            else:
                print(f"  only {len(means) - len(bad)} clusters with mean Sharpe >= 0; fallback")
            info["ExclusionFallback"] = True
    sel, status = select_stocks(score, clusters, s)
    return sel, status, info


if __name__ == "__main__":
    # 3 clusters of 4, pick 5 with <= 2 per cluster and >= 3 clusters: the best stock of the
    # strongest cluster must be skipped in favour of cluster 2's best.
    sc = np.array([10, 9, 8, 7, 5, 4, 3, 2, 1, 0.5, 0.2, 0.1])
    cl = np.repeat([0, 1, 2], 4)
    sel, _ = select_stocks(sc, cl, {"n_stocks": 5, "max_per_cluster": 2, "min_clusters": 3})
    assert sel.tolist() == [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0], sel
    print("ok")
