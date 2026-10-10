"""Run: python tests/test_pipeline.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import window_features
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

print("ok")
