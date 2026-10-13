"""Walk-forward backtest: per window, construct every strategy on training data only, freeze
it at the formation date (last training day), then buy-and-hold through the test year.

Test-year simulation (identical for every strategy):
  * Each stock grows by its corporate-action-adjusted daily return. A day with no trade
    contributes 0 (price unchanged); the next trade's return spans the gap.
  * A stock that stops trading during the test year (suspension/delisting) is held at its last
    price, i.e. as cash at 0%. The dataset has no delisting returns, so a delisting loss is
    NOT captured (documented limitation; count of such holdings is recorded).
  * No intra-year rebalancing: weights drift with prices.
  * Rebalance on the first test day. Turnover = sum_i |w_new - w_drifted_prev| (first window:
    1.0, buying from cash). Cost = cost_rate * turnover, deducted from the first day's return.

Strategies (all equal-weight unless stated; all built from the same training window):
  Proposed        K-Means + IP + GP weights
  EW_Same15       same 15 stocks, equal weight          (isolates the GP)
  MV_Same15       same 15 stocks, Markowitz max-Sharpe  (classical benchmark weighting)
  TopSharpe15     15 highest training Sharpe            (no clustering, no GP)
  TopReturn15     15 highest training Return
  Random15        15 random eligible stocks (seed + window)
  EW_Universe     every eligible stock
  NIFTY50, NIFTY500  index price returns (no costs)
Random15 is one seeded draw; random_draws.csv holds `random_draws` draws per window for a
distribution comparison.

Outputs (results/runs/<run name>/):
  windows.csv        per window: dates, universe, K, silhouette, IP/GP status & goal values,
                     scaling parameters, turnover per strategy, delisted holdings
  portfolios.csv     per (window, strategy, stock): cluster, score, weight
  daily_gross.csv    daily test-period returns before costs, per strategy
  random_draws.csv   per (window, draw): test-year return and Sharpe (gross)
"""
import copy
import sys

import numpy as np
import pandas as pd

from clustering import ZCOLS, cluster_window
from features import FEATURES
from goal_programming import goal_weights, max_sharpe_weights, train_cov
from integer_selection import select_with_quality, selection_score
from preprocessing import ROOT, load_config, load_nifty

STOCK_STRATS = ["Proposed", "EW_Same15", "MV_Same15", "TopSharpe15", "TopReturn15", "Random15",
                "EW_Universe"]
INDEX_STRATS = ["NIFTY50", "NIFTY500"]


def load_data(cfg):
    """Wide (date x stock) return matrices, cached. R = all returns, R1 = consecutive-day only
    (same convention as the Volatility/Beta features, used for the covariance)."""
    paths = {k: ROOT / v for k, v in cfg["paths"].items()}
    cache = paths["processed_dir"] / "returns_wide.parquet"
    if cache.exists():
        R = pd.read_parquet(cache)
    else:
        p = pd.read_csv(paths["processed_dir"] / "stock_panel.csv", parse_dates=["Date"],
                        usecols=["Date", "Stock", "Return", "GapDays"])
        R = p.pivot(index="Date", columns="Stock", values="Return")
        R.columns = pd.MultiIndex.from_product([["R"], R.columns])
        R1 = p[p.GapDays == 1].pivot(index="Date", columns="Stock", values="Return")
        R1.columns = pd.MultiIndex.from_product([["R1"], R1.columns])
        R = pd.concat([R, R1], axis=1)
        R.to_parquet(cache)
    nifty = load_nifty(paths["nifty50"]).set_index("Date").Return
    n500 = load_nifty(ROOT / "data/INDEX/NIFTY 500.csv").set_index("Date").Return
    feats = pd.read_csv(paths["features_dir"] / "features_all_windows.csv")
    summ = pd.read_csv(paths["features_dir"] / "window_summary.csv")
    return {"R": R["R"], "R1": R["R1"], "nifty": nifty, "n500": n500, "feats": feats, "summary": summ}


def momentum_z(feat, R, end):
    """Extension feature (robustness only): 12-month return over the last 252 trading days of
    the training window, Z-scored within the window. Uses training data only."""
    last = R.loc[:end].iloc[-252:][feat.Stock]
    mom = (1 + last.fillna(0)).prod().to_numpy() - 1
    return (mom - mom.mean()) / mom.std(ddof=0)


def construct(feat, R1_train, cfg, wid, k=None, zcols=None):
    """All strategy weights for one window from training data only.
    Returns ({strategy: Series weight by stock}, record dict, portfolio detail frame)."""
    s, g, f = cfg["selection"], cfg["goal_programming"], cfg["features"]
    feat = feat.reset_index(drop=True)
    labels, k, diag = cluster_window(feat, cfg, k=k, zcols=zcols)
    score = selection_score(feat, s["score_weights"])
    sel, ip_status, q_info = select_with_quality(score, labels, feat.Sharpe, s)
    chosen = feat[sel]
    n = s["n_stocks"]
    # GP weight bounds must admit sum w = 1 for n stocks; widen (and record) only if they cannot.
    lims = (min(g["min_weight"], 1 / n), max(g["max_weight"], 1 / n))
    if lims != (g["min_weight"], g["max_weight"]):
        print(f"  weight limits {g['min_weight']}-{g['max_weight']} infeasible for N={n}; using {lims}")
        g = {**g, "min_weight": lims[0], "max_weight": lims[1]}

    cov = train_cov(R1_train[chosen.Stock], f["trading_days"])
    mu, beta = chosen.Return.to_numpy(), chosen.Beta.to_numpy()
    w_gp, info = goal_weights(mu, beta, cov, g, f["risk_free_rate"])
    w_mv = max_sharpe_weights(mu, cov, f["risk_free_rate"], g["min_weight"], g["max_weight"])

    eq = lambda stocks: pd.Series(1 / len(stocks), index=list(stocks))
    rng = np.random.default_rng(cfg["backtest"]["random_seed"] + wid)
    W = {
        "Proposed": pd.Series(w_gp, index=chosen.Stock),
        "EW_Same15": eq(chosen.Stock),
        "MV_Same15": pd.Series(w_mv, index=chosen.Stock),
        "TopSharpe15": eq(feat.nlargest(n, "Sharpe").Stock),
        "TopReturn15": eq(feat.nlargest(n, "Return").Stock),
        "Random15": eq(rng.choice(feat.Stock, n, replace=False)),
        "EW_Universe": eq(feat.Stock),
    }
    sizes = np.bincount(labels, minlength=k)
    rec = {"N": n, "MaxPerCluster": s["max_per_cluster"], "MinClusters": s["min_clusters"],
           "MinWeight": g["min_weight"], "MaxWeight": g["max_weight"], **q_info,
           "K": k, "ClusterSizes": "|".join(map(str, sizes)),
           "ClustersSelected": labels[sel].nunique(), "MaxPerClusterSelected": labels[sel].value_counts().max(),
           "IP_Status": ip_status, "IP_Objective": score[sel].sum(), **info}
    if diag is not None:
        rec["Silhouette"] = diag.set_index("K").Silhouette[k]
        if "ExclusionFeasible" in diag:
            rec["KExclusionFeasible"] = bool(diag.set_index("K").ExclusionFeasible[k])
    detail = pd.concat([
        pd.DataFrame({"Strategy": name, "Stock": w.index, "Weight": w.to_numpy()}) for name, w in W.items()
        if name != "EW_Universe"])
    meta = feat.set_index("Stock").assign(Cluster=labels.to_numpy(), Score=score.to_numpy())
    detail = detail.join(meta[["Company", "Cluster", "Score"] + FEATURES], on="Stock")
    return W, rec, detail


def simulate(w, R_test):
    """Buy-and-hold daily returns of weights `w` over the test dates, plus end weights."""
    G = (1 + R_test[w.index].fillna(0)).cumprod()
    value = G @ w.to_numpy()
    ret = value / value.shift(fill_value=1.0) - 1
    end_w = G.iloc[-1] * w / value.iloc[-1]
    return ret, end_w


def turnover(new, old):
    return new.sub(old, fill_value=0).abs().sum()


def run(cfg, data, name="primary", k=None, momentum=False, save=True):
    w_cfg, b = cfg["windows"], cfg["backtest"]
    R, R1 = data["R"], data["R1"]
    zcols = ZCOLS + ["Z_Momentum"] if momentum else None
    prev = {s: pd.Series(dtype=float) for s in STOCK_STRATS}
    recs, details, daily, draws = [], [], [], []
    for wid, feat in data["feats"].groupby("WindowID"):
        ty = int(feat.TestYear.iat[0])
        start, end = pd.Timestamp(ty - w_cfg["train_years"], 1, 1), pd.Timestamp(ty - 1, 12, 31)
        feat = feat.copy()
        if momentum:
            feat["Z_Momentum"] = momentum_z(feat, R, end)
        W, rec, detail = construct(feat, R1.loc[start:end], cfg, wid, k=k, zcols=zcols)

        test_dates = data["nifty"].loc[str(ty)].index         # NIFTY 50 calendar of the test year
        R_test = R.reindex(test_dates)
        day = pd.DataFrame(index=test_dates)
        for sname, w in W.items():
            day[sname], end_w = simulate(w, R_test)
            rec["Turnover_" + sname] = turnover(w, prev[sname])
            prev[sname] = end_w
        day["NIFTY50"] = data["nifty"].reindex(test_dates)
        day["NIFTY500"] = data["n500"].reindex(test_dates)
        assert not day.isna().any().any(), f"missing benchmark returns in {ty}"

        # holdings that never trade again after some point of the test year (cash at 0%)
        last_trade = R_test[W["Proposed"].index].apply(lambda c: c.last_valid_index())
        rec["ProposedStoppedTrading"] = int((last_trade.isna() | (last_trade < test_dates[-21])).sum())
        rec["MaxStockDailyReturn_Proposed"] = R_test[W["Proposed"].index].max().max()

        # random-15 distribution (gross, equal weight)
        G = (1 + R_test[feat.Stock].fillna(0)).cumprod()
        rng = np.random.default_rng(b["random_seed"] * 1000 + wid)
        for d in range(b["random_draws"]):
            v = G[rng.choice(feat.Stock, cfg["selection"]["n_stocks"], replace=False)].mean(axis=1)
            r = v / v.shift(fill_value=1.0) - 1
            ann, vol = v.iloc[-1] ** (252 / len(v)) - 1, r.std() * np.sqrt(252)   # as evaluation.metrics
            draws.append({"WindowID": wid, "TestYear": ty, "Draw": d, "Return": v.iloc[-1] - 1, "AnnVol": vol,
                          "Sharpe": (ann - cfg["features"]["risk_free_rate"]) / vol})

        recs.append({"WindowID": wid, "TrainStart": start.date(), "TrainEnd": end.date(), "TestYear": ty,
                     "TestStart": test_dates[0].date(), "TestEnd": test_dates[-1].date(),
                     "Eligible": len(feat), **rec})
        details.append(detail.assign(WindowID=wid, TestYear=ty))
        daily.append(day.assign(WindowID=wid))
        print(f"[{name}] W{wid:02d} test {ty}: K={rec['K']} goals met {rec['GoalsMet']}/4 "
              f"Proposed {(1 + day.Proposed).prod() - 1:+.1%}  NIFTY50 {(1 + day.NIFTY50).prod() - 1:+.1%}")

    out = {"windows": pd.DataFrame(recs).merge(
               data["summary"].filter(regex="^(WindowID|Mean_|Std_)"), on="WindowID"),
           "portfolios": pd.concat(details), "daily_gross": pd.concat(daily).rename_axis("Date"),
           "random_draws": pd.DataFrame(draws)}
    if save:
        d = ROOT / "results/runs" / name
        d.mkdir(parents=True, exist_ok=True)
        for key, df in out.items():
            df.to_csv(d / f"{key}.csv", index=(key == "daily_gross"), float_format="%.8g")
    return out


def with_overrides(cfg, overrides):
    """Deep copy of cfg with {section: {key: value}} replaced (experiment variants)."""
    cfg = copy.deepcopy(cfg)
    for sec, kv in (overrides or {}).items():
        cfg[sec].update(kv)
    return cfg


if __name__ == "__main__":
    cfg = load_config()
    run(cfg, load_data(cfg), name=sys.argv[1] if len(sys.argv) > 1 else "primary")
