"""Build the interactive results dashboard: results/dashboard/index.html.

    python src/dashboard.py            # rebuild from the latest results and open it
    python src/dashboard.py --no-open  # rebuild only

Run after experiments.py. Every table the page shows is computed here from
results/runs/primary (+ results/clustering) and embedded
in src/dashboard_template.html together with d3 (src/vendor), so the output is one file
that works offline in any browser. Cost-dependent tables are computed for every rate in
backtest.cost_rates_sensitivity and tagged CostBps (basis points per unit turnover), so the
page's cost selector just picks rows.
"""
import json
import sys
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd

from backtest import load_data
from evaluation import STRATS, apply_costs, metrics, per_window
from preprocessing import ROOT, load_config


def rows(df):
    return json.loads(df.to_json(orient="records", date_format="iso", double_precision=6))


def build_data():
    cfg = load_config()
    data = {}
    dump = lambda name, df: data.__setitem__(name, rows(df))
    rf = cfg["features"]["risk_free_rate"]
    d = ROOT / "results/runs/primary"
    gross = pd.read_csv(d / "daily_gross.csv", parse_dates=["Date"], index_col="Date")
    win = pd.read_csv(d / "windows.csv")
    port = pd.read_csv(d / "portfolios.csv")
    m = d / "metrics"

    full, pw, weekly, ys = [], [], [], []
    for c in cfg["backtest"]["cost_rates_sensitivity"]:
        bps = int(round(c * 1e4))
        net = apply_costs(gross, win, c)
        for s in STRATS:
            to = win.get("Turnover_" + s)
            full.append({"CostBps": bps, "Strategy": s, **metrics(net[s], net.NIFTY50, rf),
                         "AvgTurnover": float(to.mean()) if to is not None else 0.0})
        pw.append(per_window(net, rf).assign(CostBps=bps))
        wealth = (1 + net[STRATS]).cumprod()
        # Proposed portfolio value at the start of each test year per 1 invested in Jan 2000
        # (before that year's rebalancing cost); the page scales it by the amount entered.
        ys.append(wealth.Proposed.shift(fill_value=1.0).groupby(wealth.index.year).first()
                  .rename_axis("TestYear").reset_index(name="Start").assign(CostBps=bps))
        dd = -(1 - wealth / wealth.cummax())
        wk = wealth.resample("W-FRI").last().join(dd.resample("W-FRI").min().add_suffix("_DD")).dropna()
        weekly.append(wk.reset_index().assign(CostBps=bps, Date=lambda x: x.Date.dt.strftime("%Y-%m-%d")))
    dump("full_period", pd.DataFrame(full))
    dump("per_window", pd.concat(pw))
    dump("wealth_weekly", pd.concat(weekly))
    dump("year_start", pd.concat(ys))

    # holdings of the proposed portfolio with each stock's realised test-year return
    R = load_data(cfg)["R"]
    h = port[port.Strategy == "Proposed"].copy()
    h["TestReturn"] = [float((1 + R.loc[str(y), s].fillna(0)).prod() - 1) for y, s in zip(h.TestYear, h.Stock)]
    h["Contribution"] = h.Weight * h.TestReturn
    h["Key"] = h.TestYear.astype(str) + "-" + h.Stock
    dump("holdings", h.drop(columns="Strategy"))

    cols = ["WindowID", "TrainStart", "TrainEnd", "TestYear", "TestStart", "TestEnd", "Eligible", "K",
            "Silhouette", "ClusterSizes", "GoalsMet"] + [c for c in win if c.startswith(("GP_", "Dev_", "Turnover_"))] \
        + ["ProposedStoppedTrading"]
    dump("windows", win[[c for c in cols if c != "GP_Status"]])
    prof = pd.read_csv(ROOT / "results/clustering/cluster_profiles.csv")
    dump("cluster_profiles", prof.assign(Key=prof.TestYear.astype(str) + "-" + prof.Cluster.astype(str)))
    kd = pd.read_csv(ROOT / "results/clustering/k_diagnostics.csv")
    dump("k_diagnostics", kd.assign(Key=kd.TestYear.astype(str) + "-" + kd.K.astype(str)))
    dump("random_percentile", pd.read_csv(m / "random_percentile.csv"))
    return data


def main(open_browser=True):
    data = build_data()
    html = (ROOT / "src/dashboard_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")   # never closes the <script>
    html = (html.replace("/*__D3__*/", (ROOT / "src/vendor/d3.v7.9.0.min.js").read_text(encoding="utf-8"))
                .replace("/*__DATA__*/", payload)
                .replace("/*__BUILT__*/", datetime.now().strftime("%Y-%m-%d %H:%M")))
    out = ROOT / "results/dashboard/index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"dashboard: {out} ({out.stat().st_size / 1e6:.1f} MB, {sum(map(len, data.values()))} rows)")
    if open_browser:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main(open_browser="--no-open" not in sys.argv)
