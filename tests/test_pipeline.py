"""Run: python tests/test_pipeline.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import window_features
from backtest import simulate, turnover
from evaluation import apply_costs, metrics
from preprocessing import ca_factor, flag_spikes, load_config

cfg = load_config()

# 1:1 bonus (0.5), 1:10 split (0.1), genuine -40% crash, 2:1 consolidation, bonus after long gap
f = ca_factor(pd.Series([0.5, 0.1, 0.6, 2.0, 0.5]), pd.Series([1, 1, 1, 1, 30]), cfg["preprocessing"])
assert np.allclose(f, [2, 10, 1, 0.5, 1]), f

# 2-day bad print 100 -> 150 -> 150 -> 101 is dropped; a real +30% that stays is kept
close = pd.Series([100, 150, 150, 101, 102, 100, 130, 131.0])
stock = pd.Series(["A"] * 5 + ["B"] * 3)
assert flag_spikes(close, stock, 0.25, 0.05, 5).tolist() == [False, True, True] + [False] * 5

# stock = exactly 2x the market every day -> beta 2; known drawdown
dates = pd.bdate_range("2020-01-01", periods=5)
mkt = pd.Series([0.0, 0.01, -0.02, 0.03, -0.01], index=dates)
r = 2 * mkt
sub = pd.DataFrame({"Stock": "A", "Date": dates, "Close": 100 * (1 + r).cumprod(), "Return": r,
                    "GapDays": 1.0, "TradedValue": 1e6})
out = window_features(sub, mkt, len(dates), cfg["features"]).iloc[0]
idx = (1 + r.iloc[1:]).cumprod()
assert np.isclose(out.Beta, 2), out.Beta
assert np.isclose(out.MaxDrawdown, (1 - idx / idx.cummax()).max())
assert np.isclose(out.Liquidity, np.log1p(1e6))

# buy-and-hold: A +10% then flat; B no trade day 1 (NaN = unchanged), then +20%
Rt = pd.DataFrame({"A": [0.10, 0.0], "B": [np.nan, 0.20]}, index=dates[:2])
ret, end_w = simulate(pd.Series({"A": 0.5, "B": 0.5}), Rt)
assert np.allclose(ret, [0.05, 0.1 / 1.05]) and np.isclose(end_w.sum(), 1)
assert np.isclose((1 + ret).prod(), 0.5 * 1.1 + 0.5 * 1.2)            # = 1.15
assert np.isclose(turnover(pd.Series({"A": 1.0}), pd.Series({"B": 1.0})), 2)  # full switch

# costs hit only the first day of each window, scaled by that window's turnover
from backtest import STOCK_STRATS
g = pd.DataFrame({s: [0.0, 0.0] for s in STOCK_STRATS + ["NIFTY50", "NIFTY500"]}, index=dates[:2]).assign(WindowID=1)
w = pd.DataFrame({"WindowID": [1], **{"Turnover_" + s: [2.0] for s in STOCK_STRATS}})
n = apply_costs(g, w, 0.01)
assert np.allclose(n.Proposed, [-0.02, 0]) and (n.NIFTY50 == 0).all()

m = metrics(pd.Series([0.1, -0.5, 0.2]), pd.Series([0.1, -0.5, 0.2]), 0.0)
assert np.isclose(m["MaxDrawdown"], 0.5) and np.isclose(m["Beta"], 1) and np.isclose(m["CumReturn"], 1.1 * 0.5 * 1.2 - 1)
print("ok")
