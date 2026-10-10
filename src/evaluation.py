"""Out-of-sample evaluation of a backtest run (results/runs/<run>/).

Everything here is computed on TEST-YEAR returns only; training-period figures (GP goal
values etc.) live in windows.csv and are never reported as performance.

Metric definitions (daily returns r, rf = constant annual risk-free rate):
  CumReturn   prod(1 + r) - 1
  AnnReturn   (1 + CumReturn)^(252 / n_days) - 1           (geometric, as the Return feature)
  AnnVol      std(r) * sqrt(252)
  Sharpe      (AnnReturn - rf) / AnnVol                     (same form as the Sharpe feature)
  Sortino     (AnnReturn - rf) / downside deviation,  DD = sqrt(mean(min(r - rf/252, 0)^2)) * sqrt(252)
  MaxDrawdown largest peak-to-trough fall of the wealth curve (positive magnitude)
  Beta        cov(r, NIFTY50) / var(NIFTY50) on test days
Costs: net = gross with (1 - cost_rate * turnover) applied on each window's first day.

Statistics: paired per-window comparisons of Proposed vs every other strategy
(wins, mean difference, Wilcoxon signed-rank p-value, bootstrap 95% CI of the mean).
Regimes (analysis only, ex-post labels never used for construction): test year is Bull if
NIFTY 50 rose that year, else Bear; High-vol if NIFTY 50 test-year vol is above the median
across test years.

Outputs (results/runs/<run>/metrics/): per_window.csv, full_period.csv, comparisons.csv,
random_percentile.csv, regimes.csv, cost_sensitivity.csv; plots in results/runs/<run>/plots/.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from backtest import INDEX_STRATS, STOCK_STRATS
from preprocessing import ROOT, load_config

STRATS = STOCK_STRATS + INDEX_STRATS


def apply_costs(gross, windows, rate):
    net = gross.copy()
    first = ~net.WindowID.duplicated()
    to = windows.set_index("WindowID")
    for s in STOCK_STRATS:
        c = rate * net.loc[first, "WindowID"].map(to["Turnover_" + s])
        net.loc[first, s] = (1 + net.loc[first, s]) * (1 - c) - 1
    return net


def metrics(r, mkt, rf):
    wealth = (1 + r).cumprod()
    cum = wealth.iloc[-1] - 1
    ann = (1 + cum) ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    dd = np.sqrt((np.minimum(r - rf / 252, 0) ** 2).mean()) * np.sqrt(252)
    return {"CumReturn": cum, "AnnReturn": ann, "AnnVol": vol, "Sharpe": (ann - rf) / vol,
            "Sortino": (ann - rf) / dd, "MaxDrawdown": (1 - wealth / wealth.cummax()).max(),
            "Beta": np.cov(r, mkt)[0, 1] / mkt.var()}


def per_window(net, rf):
    rows = [{"WindowID": wid, "TestYear": g.index[0].year, "Strategy": s, **metrics(g[s], g.NIFTY50, rf)}
            for wid, g in net.groupby("WindowID") for s in STRATS]
    return pd.DataFrame(rows)


def full_period(net, gross, windows, rf):
    out = []
    for s in STRATS:
        m = metrics(net[s], net.NIFTY50, rf)
        m["GrossAnnReturn"] = metrics(gross[s], gross.NIFTY50, rf)["AnnReturn"]
        to = windows.get("Turnover_" + s)
        m["AvgTurnover"] = to.mean() if to is not None else 0.0
        out.append({"Strategy": s, **m})
    return pd.DataFrame(out)


def bootstrap_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def comparisons(pw):
    piv = {m: pw.pivot(index="WindowID", columns="Strategy", values=m) for m in ["AnnReturn", "Sharpe", "MaxDrawdown"]}
    rows = []
    for s in STRATS[1:]:
        for m, p in piv.items():
            d = p["Proposed"] - p[s]
            if m == "MaxDrawdown":
                d = -d                                   # positive = Proposed fell less
            lo, hi = bootstrap_ci(d.to_numpy())
            rows.append({"Baseline": s, "Metric": m, "Periods": len(d), "ProposedWins": int((d > 0).sum()),
                         "MeanDiff": d.mean(), "MedianDiff": d.median(), "CI95_Low": lo, "CI95_High": hi,
                         "WilcoxonP": wilcoxon(d).pvalue})
    return pd.DataFrame(rows)


def random_percentile(pw_gross, draws):
    """Where the Proposed portfolio's GROSS test-year return/Sharpe ranks among random 15s."""
    prop = pw_gross[pw_gross.Strategy == "Proposed"].set_index("WindowID")
    rows = []
    for wid, d in draws.groupby("WindowID"):
        p = prop.loc[wid]
        rows.append({"WindowID": wid, "TestYear": p.TestYear,
                     "ReturnPctile": (d.Return < p.CumReturn).mean(),
                     "SharpePctile": (d.Sharpe < p.Sharpe).mean(),
                     "RandomMedianReturn": d.Return.median(), "ProposedReturn": p.CumReturn})
    return pd.DataFrame(rows)


def regimes(pw, net):
    n50 = pw[pw.Strategy == "NIFTY50"].set_index("WindowID")
    vol_med = n50.AnnVol.median()
    lab = pd.DataFrame({"Trend": np.where(n50.CumReturn > 0, "Bull", "Bear"),
                        "Vol": np.where(n50.AnnVol > vol_med, "HighVol", "LowVol")}, index=n50.index)
    x = pw.join(lab, on="WindowID")
    out = []
    for col in ["Trend", "Vol"]:
        g = x.groupby([col, "Strategy"]).agg(Years=("TestYear", "size"), MeanAnnReturn=("AnnReturn", "mean"),
                                             MeanSharpe=("Sharpe", "mean"), MeanMaxDD=("MaxDrawdown", "mean"))
        out.append(g.reset_index().rename(columns={col: "Regime"}).assign(Dimension=col))
    return pd.concat(out), lab


def cost_sensitivity(gross, windows, rates, rf):
    rows = []
    for c in rates:
        net = apply_costs(gross, windows, c)
        for s in ["Proposed", "EW_Same15", "MV_Same15", "TopSharpe15", "TopReturn15", "Random15", "EW_Universe", "NIFTY50"]:
            m = metrics(net[s], net.NIFTY50, rf)
            rows.append({"CostRate": c, "Strategy": s, "AnnReturn": m["AnnReturn"], "Sharpe": m["Sharpe"]})
    return pd.DataFrame(rows)


def plots(net, pw, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#0b0b0b", "#898781", "#e1e0d9"
    colors = {"Proposed": "#2a78d6", "EW_Same15": "#eb6834", "NIFTY50": "#1baf7a", "NIFTY500": "#eda100"}
    plt.rcParams.update({"axes.edgecolor": "#c3c2b7", "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "font.size": 10})
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for s, c in colors.items():
        w = (1 + net[s]).cumprod()
        ax.plot(w.index, w, color=c, lw=2, label=f"{s} ({w.iloc[-1]:.1f}x)")
    ax.set_yscale("log")
    ax.set_title("Growth of 1 INR, out-of-sample 2000 - Jun 2021 (net of costs, log scale)", loc="left", color=ink)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "cumulative_wealth.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for s, c in colors.items():
        w = (1 + net[s]).cumprod()
        ax.plot(w.index, -(1 - w / w.cummax()), color=c, lw=1.5, label=s)
    ax.set_title("Drawdown from running peak (net)", loc="left", color=ink)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.legend(frameon=False, loc="lower left", ncol=4)
    fig.tight_layout(); fig.savefig(out / "drawdowns.png", dpi=150); plt.close(fig)

    p = pw.pivot(index="TestYear", columns="Strategy", values="CumReturn")
    fig, ax = plt.subplots(figsize=(11, 4))
    x, bw = np.arange(len(p)), 0.38
    ax.bar(x - bw / 2 - 0.01, p.Proposed, bw, color=colors["Proposed"], label="Proposed")
    ax.bar(x + bw / 2 + 0.01, p.NIFTY50, bw, color=colors["NIFTY50"], label="NIFTY50")
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xticks(x, [str(y) if y < 2021 else "2021*" for y in p.index], rotation=45)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_title("Test-year return, Proposed vs NIFTY 50 (net; *2021 = Jan-Jun)", loc="left", color=ink)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "annual_returns.png", dpi=150); plt.close(fig)


def evaluate(run, cfg, cost_rate=None, make_plots=True):
    d = ROOT / "results/runs" / run
    rf, b = cfg["features"]["risk_free_rate"], cfg["backtest"]
    cost_rate = b["cost_rate"] if cost_rate is None else cost_rate
    gross = pd.read_csv(d / "daily_gross.csv", parse_dates=["Date"], index_col="Date")
    windows = pd.read_csv(d / "windows.csv")
    draws = pd.read_csv(d / "random_draws.csv")
    net = apply_costs(gross, windows, cost_rate)

    pw = per_window(net, rf)
    res = {"per_window": pw, "full_period": full_period(net, gross, windows, rf),
           "comparisons": comparisons(pw),
           "random_percentile": random_percentile(per_window(gross, rf), draws),
           "cost_sensitivity": cost_sensitivity(gross, windows, b["cost_rates_sensitivity"], rf)}
    res["regimes"], labels = regimes(pw, net)
    m = d / "metrics"
    m.mkdir(exist_ok=True)
    for k, v in res.items():
        v.to_csv(m / f"{k}.csv", index=False, float_format="%.6g")
    labels.to_csv(m / "regime_labels.csv")
    if make_plots:
        plots(net, pw, d / "plots")
    return res


if __name__ == "__main__":
    cfg = load_config()
    res = evaluate(sys.argv[1] if len(sys.argv) > 1 else "primary", cfg)
    pd.set_option("display.width", 200)
    print(res["full_period"].round(3).to_string(index=False))
