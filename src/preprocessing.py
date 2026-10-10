"""Load, clean and align NSE stock data with the NIFTY 50 calendar.

Outputs (data/processed/):
  nifty50_clean.csv           Date, Close, Return
  stock_panel.csv             Date, Stock, Close, Volume, TradedValue, Return, GapDays
  corporate_actions.csv       every detected split/bonus/consolidation (audit log)
  preprocessing_report.csv    per-stock row counts removed at each step

Key facts about the raw data (verified by inspection, see README in summary):
  * Prices are NOT corporate-action adjusted and `Prev Close` is not adjusted on ex-dates.
  * `Turnover` is scaled by 1e5 vs Close*Volume, so traded value is recomputed here.
  * The file name is the stable stock id; the `Symbol` column changes on renames.
  * Missing dates are simply absent rows (non-trading / pre-listing / post-delisting);
    nothing is forward-filled.

No step uses information from after the row's own date, so preprocessing cannot leak
test-year data into training windows. Returns are adjusted per event, never by
back-propagating future adjustment factors through earlier prices.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    return yaml.safe_load((ROOT / "configs/parameters.yaml").read_text())


def load_nifty(path):
    n = pd.read_csv(path, usecols=["Date", "Close"], parse_dates=["Date"])
    n = n.drop_duplicates("Date").query("Close > 0").sort_values("Date").reset_index(drop=True)
    n["Return"] = n.Close.pct_change()
    return n


def load_stocks(scrip_dir):
    cols = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
    frames = [pd.read_csv(f, usecols=cols).assign(Stock=f.stem) for f in sorted(Path(scrip_dir).glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df[cols[2:]] = df[cols[2:]].astype(float)   # empty files make concat fall back to object
    return df.sort_values(["Stock", "Date"], kind="stable").reset_index(drop=True)


def flag_spikes(close, stock, min_move, return_tol, max_len):
    """Bad prints lasting 1..max_len rows: a big move away, then a big opposite move that brings
    the price back within return_tol of where it started. All rows of the excursion are flagged."""
    g = close.groupby(stock)
    lr = np.log(close / g.shift())
    big = np.log1p(min_move)
    flag = pd.Series(False, index=close.index)
    for k in range(1, max_len + 1):
        lr_back = lr.groupby(stock).shift(-k)                 # move on the row k steps later
        back = np.log(g.shift(-k) / g.shift())                # price k rows later vs pre-excursion
        start = (lr.abs() > big) & (lr_back.abs() > big) & (np.sign(lr) != np.sign(lr_back)) \
            & (back.abs() < np.log1p(return_tol))
        for j in range(k):                                    # rows t .. t+k-1 are the bad prints
            flag |= start.groupby(stock).shift(j, fill_value=False)
    return flag


def ca_factor(ratio, gap, cfg):
    """Price-multiplier that neutralises a split/bonus (f>1) or consolidation (f<1); 1 if none.

    ratio = Close_t / Close_{t-1} (raw). A 1:1 bonus gives ratio ~0.5 -> factor 2.
    Only jumps across <= ca_max_gap NIFTY days qualify: a jump after a long suspension is
    treated as a genuine price move. Consolidations must be >= 2:1 (3:2 reverse splits are
    practically unknown on NSE; ~1.5x up-jumps were penny-stock tick noise).
    ponytail: ratio-matching heuristic; a genuine crash that happens to land on a standard
    ratio is misread as a corporate action. Replace with an NSE corporate-actions file if one
    becomes available. Demergers/rights issues (non-standard ratios) are not adjusted.
    """
    f = np.asarray(cfg["ca_factors"], dtype=float)
    r = np.asarray(ratio, dtype=float)[:, None]
    ok = np.asarray(gap <= cfg["ca_max_gap"])
    out = np.ones(len(r))
    for cand, mask in ((f, r[:, 0] <= cfg["ca_max_ratio"]), (1 / f[f >= 2], r[:, 0] >= 1.8)):
        err = np.abs(r * cand - 1)
        best = err.argmin(axis=1)
        hit = ok & mask & (err[np.arange(len(r)), best] <= cfg["ca_tol"])
        out[hit] = cand[best[hit]]
    return out


def clean(df, cal, cfg):
    """Return (panel, corporate_actions, report). `cal` = NIFTY trading dates (sorted)."""
    p = cfg["preprocessing"]
    rep = df.groupby("Stock").agg(Symbols=("Symbol", lambda s: "|".join(s.unique())), RawRows=("Date", "size"))

    def drop(mask, name):
        nonlocal df
        rep[name] = df[mask].groupby("Stock").size().reindex(rep.index, fill_value=0)
        df = df[~mask]

    drop(df.duplicated(["Stock", "Date"]), "DuplicatesRemoved")          # identical re-prints
    drop(~df.Date.isin(cal), "OffCalendarRemoved")                         # dates NIFTY did not trade
    ohl_bad = (df[["Open", "High", "Low"]] <= 0).any(axis=1) | (df.High < df.Low) \
        | (df.Close > df.High * 1.001) | (df.Close < df.Low * 0.999)
    drop(df.Close.isna() | (df.Close <= 0) | df.Volume.isna() | (df.Volume <= 0) | ohl_bad, "InvalidRemoved")
    df = df.reset_index(drop=True)
    drop(flag_spikes(df.Close, df.Stock, p["spike_min_move"], p["spike_return_tol"], p["spike_max_len"]),
         "SpikesRemoved")

    df = df.reset_index(drop=True)
    pos = pd.Series(np.arange(len(cal)), index=cal)
    df["GapDays"] = df.Date.map(pos).groupby(df.Stock).diff()   # 1 = consecutive NIFTY days
    prev = df.groupby("Stock").Close.shift()
    ratio = df.Close / prev
    factor = ca_factor(ratio.fillna(1), df.GapDays.fillna(np.inf), p)
    df["Return"] = ratio * factor - 1          # NaN on each stock's first row
    df["TradedValue"] = df.Close * df.Volume                     # INR

    is_ca = factor != 1
    ca = df.loc[is_ca, ["Stock", "Date", "Close", "Return"]].assign(
        PrevClose=prev[is_ca], RawRatio=ratio[is_ca], Factor=factor[is_ca],
        Type=np.where(factor[is_ca] > 1, "split/bonus", "consolidation"))
    rep["CorporateActions"] = ca.groupby("Stock").size().reindex(rep.index, fill_value=0)
    g = df.groupby("Stock").Date
    rep["FinalRows"], rep["FirstDate"], rep["LastDate"] = g.size(), g.min(), g.max()
    panel = df[["Date", "Stock", "Close", "Volume", "TradedValue", "Return", "GapDays"]]
    return panel, ca, rep.reset_index()


def main():
    cfg = load_config()
    paths = {k: ROOT / v for k, v in cfg["paths"].items()}
    out = paths["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)

    nifty = load_nifty(paths["nifty50"])
    raw = load_stocks(paths["scrip_dir"])
    panel, ca, rep = clean(raw, nifty.Date.values, cfg)
    names = pd.read_csv(paths["symbols"]).set_index("Scrip")["Company Name"]
    rep.insert(1, "Company", rep.Stock.map(names))

    nifty.to_csv(out / "nifty50_clean.csv", index=False)
    panel.to_csv(out / "stock_panel.csv", index=False, float_format="%.8g")
    ca.to_csv(out / "corporate_actions.csv", index=False)
    rep.to_csv(out / "preprocessing_report.csv", index=False)

    removed = rep[["DuplicatesRemoved", "OffCalendarRemoved", "InvalidRemoved", "SpikesRemoved"]].sum()
    print(f"NIFTY 50: {len(nifty)} days {nifty.Date.min().date()}..{nifty.Date.max().date()}")
    print(f"Stocks: {rep.Stock.nunique()}  raw rows: {len(raw)}  final rows: {len(panel)}")
    print("Rows removed:", removed.to_dict())
    print(f"Corporate actions detected: {len(ca)} ({ca.Type.value_counts().to_dict()})")


if __name__ == "__main__":
    main()
