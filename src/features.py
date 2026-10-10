"""Per-window investable universe + the six behavioural features + window-level Z-scores.

For test year Y the training window is 1 Jan (Y-4) .. 31 Dec (Y-1). Everything here uses
only rows inside that window, so nothing from year Y (or later) touches the features or the
scaling parameters.

Feature definitions (identical in every window):
  Return      annualised geometric return (CAGR) of the corporate-action-adjusted return
              index from the stock's first to last observation in the window,
              (prod(1+r))^(252 / NIFTY days spanned) - 1. Price returns (no dividends in data).
  Volatility  std(daily returns) * sqrt(252), using only returns between consecutive NIFTY
              trading days (GapDays == 1) so multi-day returns don't inflate it.
  Sharpe      (Return - rf) / Volatility, rf = constant (configs/parameters.yaml).
  Beta        cov(r_i, r_NIFTY50) / var(r_NIFTY50) on the same consecutive-day dates.
  MaxDrawdown largest peak-to-trough fall of the adjusted return index, stored as a POSITIVE
              loss magnitude (0.6 = -60%), so higher = worse, like Volatility.
  Liquidity   log(1 + average daily traded value in INR), average over ALL NIFTY trading days
              in the window (no-trade days count as zero). log because traded value is
              heavily right-skewed.

Z-scores: (x - window mean) / window std (ddof=0, same as sklearn StandardScaler) over the
eligible stocks of that window only.

Outputs (results/features/):
  features_all_windows.csv  one row per (window, eligible stock): raw + Z features
  window_summary.csv        one row per window: dates, universe counts, scaling mean/std
"""
import numpy as np
import pandas as pd

from preprocessing import ROOT, load_config

FEATURES = ["Return", "Volatility", "Sharpe", "Beta", "MaxDrawdown", "Liquidity"]


def eligibility(stats, cal_w, u):
    """Investable at the formation date (last training day), using training data only.
    A stock is NOT required to survive into the test year (that would be look-ahead)."""
    return pd.DataFrame({
        "LowHistory": stats.NObs < u["min_obs_frac"] * len(cal_w),
        "NotTradingAtFormation": stats.LastDate < cal_w[-u["last_obs_within_days"]],
        "PennyStock": stats.LastClose < u["min_price"],
    })


def window_features(sub, nifty_ret, n_days, f):
    """Six raw features per stock. `sub` = panel rows of eligible stocks inside the window,
    sorted by Stock, Date. The first row of each stock carries a return from before the
    window, so it is dropped from all return-based calculations."""
    sub = sub.copy()
    first = sub.groupby("Stock").cumcount() == 0
    sub.loc[first, ["Return", "GapDays"]] = np.nan
    sub["LogRet"] = np.log1p(sub.Return).fillna(0.0)
    g = sub.groupby("Stock")

    out = pd.DataFrame({"NObs": g.size(), "FirstDate": g.Date.min(), "LastDate": g.Date.max(),
                        "LastClose": g.Close.last()})
    out["Return"] = np.exp(g.LogRet.sum() * f["trading_days"] / g.GapDays.sum()) - 1

    idx = np.exp(g.LogRet.cumsum())                               # adjusted index, 1 at first obs
    dd = 1 - idx / idx.groupby(sub.Stock).cummax()
    out["MaxDrawdown"] = dd.groupby(sub.Stock).max()

    d = sub[sub.GapDays == 1].assign(Mkt=lambda x: x.Date.map(nifty_ret))
    gd = d.groupby("Stock")
    out["Volatility"] = gd.Return.std() * np.sqrt(f["trading_days"])
    d = d.assign(XY=d.Return * d.Mkt, YY=d.Mkt ** 2)
    m = d.groupby("Stock")[["Return", "Mkt", "XY", "YY"]].mean()
    out["Beta"] = (m.XY - m.Return * m.Mkt) / (m.YY - m.Mkt ** 2)
    out["Sharpe"] = (out.Return - f["risk_free_rate"]) / out.Volatility

    out["AvgTradedValue"] = g.TradedValue.sum() / n_days
    out["Liquidity"] = np.log1p(out.AvgTradedValue)
    return out


def main():
    cfg = load_config()
    paths = {k: ROOT / v for k, v in cfg["paths"].items()}
    w, u, f = cfg["windows"], cfg["universe"], cfg["features"]
    nifty = pd.read_csv(paths["processed_dir"] / "nifty50_clean.csv", parse_dates=["Date"])
    panel = pd.read_csv(paths["processed_dir"] / "stock_panel.csv", parse_dates=["Date"])
    names = pd.read_csv(paths["processed_dir"] / "preprocessing_report.csv").set_index("Stock").Company
    nifty_ret = nifty.set_index("Date").Return

    rows, summary = [], []
    for wid, test_year in enumerate(range(w["first_test_year"], w["last_test_year"] + 1), start=1):
        start, end = pd.Timestamp(test_year - w["train_years"], 1, 1), pd.Timestamp(test_year - 1, 12, 31)
        cal_w = nifty.Date[nifty.Date.between(start, end)].values
        sub = panel[panel.Date.between(start, end)]            # panel is sorted by Stock, Date
        g = sub.groupby("Stock")
        stats = pd.DataFrame({"NObs": g.size(), "LastDate": g.Date.max(), "LastClose": g.Close.last()})
        excl = eligibility(stats, cal_w, u)
        keep = excl.index[~excl.any(axis=1)]

        feat = window_features(sub[sub.Stock.isin(keep)], nifty_ret, len(cal_w), f)
        bad = feat[FEATURES].isna().any(axis=1)                # e.g. zero-variance stock
        feat = feat[~bad]
        mu, sd = feat[FEATURES].mean(), feat[FEATURES].std(ddof=0)
        for c in FEATURES:
            feat["Z_" + c] = (feat[c] - mu[c]) / sd[c]

        meta = {"WindowID": wid, "TrainStart": start.date(), "TrainEnd": end.date(), "TestYear": test_year}
        feat = feat.reset_index().assign(**meta, Company=lambda x: x.Stock.map(names))
        rows.append(feat)
        summary.append({**meta, "NiftyTradingDays": len(cal_w), "StocksWithData": len(stats),
                        **{"Excl_" + k: int(v) for k, v in excl.sum().items()},
                        "Excl_FeatureNaN": int(bad.sum()), "Eligible": len(feat),
                        **{"Mean_" + c: mu[c] for c in FEATURES}, **{"Std_" + c: sd[c] for c in FEATURES}})
        print(f"W{wid:02d} train {start.year}-{end.year} test {test_year}: "
              f"{len(stats)} with data -> {len(feat)} eligible")

    cols = ["WindowID", "TrainStart", "TrainEnd", "TestYear", "Stock", "Company", "NObs", "FirstDate",
            "LastDate", "LastClose", "AvgTradedValue"] + FEATURES + ["Z_" + c for c in FEATURES]
    out = paths["features_dir"]
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows)[cols].to_csv(out / "features_all_windows.csv", index=False, float_format="%.6g")
    pd.DataFrame(summary).to_csv(out / "window_summary.csv", index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
