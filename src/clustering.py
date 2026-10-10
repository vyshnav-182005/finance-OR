"""K-Means behavioural clustering on the window's Z-scored six features.

Purpose is diversification (which stocks behave alike), not prediction. Everything uses
only the training-window feature matrix that features.py produced; K is chosen per window
from training data only.

Outputs (results/clustering/) when run as a script:
  k_diagnostics.csv        per (window, K): silhouette, inertia, cluster sizes, IP-feasible
  cluster_assignments.csv  per (window, stock): chosen K and cluster label
  cluster_profiles.csv     per (window, cluster): size, mean raw/Z features, text description
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from features import FEATURES
from preprocessing import ROOT, load_config

ZCOLS = ["Z_" + c for c in FEATURES]


def fit_kmeans(Z, k, c):
    return KMeans(n_clusters=k, n_init=c["n_init"], random_state=c["seed"]).fit(Z)


def ip_feasible(labels, k, s):
    """Can the IP pick n_stocks with <= max_per_cluster per cluster and >= min_clusters clusters?"""
    sizes = np.bincount(labels, minlength=k)
    return k >= s["min_clusters"] and np.minimum(sizes, s["max_per_cluster"]).sum() >= s["n_stocks"]


def k_diagnostics(Z, c, s):
    rows, models = [], {}
    for k in c["k_candidates"]:
        km = fit_kmeans(Z, k, c)
        sizes = np.bincount(km.labels_, minlength=k)
        rows.append({"K": k, "Silhouette": silhouette_score(Z, km.labels_), "Inertia": km.inertia_,
                     "MinSize": sizes.min(), "MaxSize": sizes.max(), "Sizes": "|".join(map(str, sizes)),
                     "IPFeasible": ip_feasible(km.labels_, k, s)})
        models[k] = km
    return pd.DataFrame(rows), models


def choose_k(diag):
    """Max silhouette among IP-feasible K. Training data only."""
    ok = diag[diag.IPFeasible]
    if ok.empty:
        raise ValueError("no IP-feasible K among candidates")
    return int(ok.loc[ok.Silhouette.idxmax(), "K"])


def cluster_window(feat, cfg, k=None, zcols=None):
    """Return (labels Series indexed like feat, chosen K, diagnostics or None).
    k given = fixed-K experiment (still must be IP-feasible, else error).
    zcols = clustering feature columns (default: the six core Z features)."""
    c, s = cfg["clustering"], cfg["selection"]
    Z = feat[zcols or ZCOLS].to_numpy()
    if k is None:
        diag, models = k_diagnostics(Z, c, s)
        k = choose_k(diag)
        km = models[k]
    else:
        diag, km = None, fit_kmeans(Z, k, c)
        if not ip_feasible(km.labels_, k, s):
            raise ValueError(f"fixed K={k} is not IP-feasible")
    return pd.Series(km.labels_, index=feat.index, name="Cluster"), k, diag


def describe(zmeans, thr=0.5):
    """Cluster numbers are arbitrary: name a cluster by its above/below-average features."""
    parts = [("high " if v > 0 else "low ") + f for f, v in zmeans.items() if abs(v) >= thr]
    return ", ".join(parts) or "near average"


def profile(feat, labels):
    g = feat.assign(Cluster=labels).groupby("Cluster")
    prof = g[FEATURES + ZCOLS].mean()
    prof.insert(0, "Size", g.size())
    prof["Description"] = [describe(r[ZCOLS].rename(lambda x: x[2:])) for _, r in prof.iterrows()]
    return prof.reset_index()


def main():
    cfg = load_config()
    feats = pd.read_csv(ROOT / cfg["paths"]["features_dir"] / "features_all_windows.csv")
    diags, assigns, profs = [], [], []
    for wid, feat in feats.groupby("WindowID"):
        labels, k, diag = cluster_window(feat, cfg)
        meta = {"WindowID": wid, "TestYear": feat.TestYear.iat[0]}
        diags.append(diag.assign(**meta, Chosen=diag.K == k))
        assigns.append(feat[["Stock"]].assign(**meta, K=k, Cluster=labels))
        profs.append(profile(feat, labels).assign(**meta, K=k))
        d = diag.set_index("K").loc[k]
        print(f"W{wid:02d} test {meta['TestYear']}: K={k} sil={d.Silhouette:.3f} sizes={d.Sizes}")
    out = ROOT / "results/clustering"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(diags).to_csv(out / "k_diagnostics.csv", index=False, float_format="%.6g")
    pd.concat(assigns).to_csv(out / "cluster_assignments.csv", index=False)
    pd.concat(profs).to_csv(out / "cluster_profiles.csv", index=False, float_format="%.4g")


if __name__ == "__main__":
    main()
