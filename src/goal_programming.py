"""Goal Programming: HOW MUCH in each IP-selected stock.

Weighted GP over the n selected stocks. Variables: weights w and unwanted deviations
d = (d_R, d_V, d_S, d_B) >= 0.

    min  sum_k p_k * d_k / |T_k|  +  eps * sigma(w)
    s.t. mu'w + d_R           >= T_Return       (return at least target)
         sigma(w) - d_V       <= T_Volatility   (volatility at most target)
         Sharpe(w) + d_S      >= T_Sharpe       Sharpe(w) = (mu'w - rf) / sigma(w)
         beta'w - d_B         <= T_Beta
         sum w = 1,  min_weight <= w_i <= max_weight

mu_i = stock's training-window annualised return feature (CAGR); beta_i = training beta;
sigma(w) = sqrt(w' Sigma w), Sigma = annualised covariance of training daily returns, so
portfolio risk includes co-movement between the selected stocks. mu'w approximates the
portfolio's annual return with the weighted CAGR (a slight underestimate of the
rebalanced portfolio's geometric return).
eps (1e-3) only breaks ties when every goal is met: then the least-risk weighting wins.
Solved with SLSQP (scipy) from equal weights.
"""
import numpy as np
from scipy.optimize import minimize

GOALS = ["Return", "Volatility", "Sharpe", "Beta"]
EPS = 1e-3


def train_cov(R, trading_days):
    """Annualised covariance from a (dates x stocks) training return frame (NaN = no
    consecutive-day return). Pairwise-complete estimates can be indefinite, so negative
    eigenvalues are clipped to a small positive floor."""
    S = R.cov(min_periods=60).to_numpy() * trading_days
    if np.isnan(S).any():
        raise ValueError("covariance has pairs with < 60 common observations")
    vals, vecs = np.linalg.eigh(S)
    return (vecs * np.clip(vals, 1e-8, None)) @ vecs.T


def portfolio_stats(w, mu, beta, cov, rf):
    vol = np.sqrt(w @ cov @ w)
    ret = w @ mu
    return {"Return": ret, "Volatility": vol, "Sharpe": (ret - rf) / vol, "Beta": w @ beta}


def goal_weights(mu, beta, cov, g, rf):
    """Return (weights, info). info holds achieved goal values, deviations, solver status."""
    n = len(mu)
    T, P = g["targets"], g["priorities"]
    pk = np.array([P[k] / abs(T[k]) for k in GOALS])
    sig = lambda w: np.sqrt(w @ cov @ w)
    dsig = lambda w: cov @ w / sig(w)                      # analytic gradients: finite
    e = lambda j: np.eye(4)[j]                             # differences made SLSQP stall
    obj = lambda z: pk @ z[n:] + EPS * sig(z[:n])
    obj_jac = lambda z: np.r_[EPS * dsig(z[:n]), pk]
    sharpe = lambda w: (w @ mu - rf) / sig(w)
    dsharpe = lambda w: mu / sig(w) - (w @ mu - rf) * (cov @ w) / sig(w) ** 3
    cons = [
        {"type": "eq", "fun": lambda z: z[:n].sum() - 1, "jac": lambda z: np.r_[np.ones(n), np.zeros(4)]},
        {"type": "ineq", "fun": lambda z: z[:n] @ mu + z[n] - T["Return"], "jac": lambda z: np.r_[mu, e(0)]},
        {"type": "ineq", "fun": lambda z: T["Volatility"] + z[n + 1] - sig(z[:n]),
         "jac": lambda z: np.r_[-dsig(z[:n]), e(1)]},
        {"type": "ineq", "fun": lambda z: sharpe(z[:n]) + z[n + 2] - T["Sharpe"],
         "jac": lambda z: np.r_[dsharpe(z[:n]), e(2)]},
        {"type": "ineq", "fun": lambda z: T["Beta"] + z[n + 3] - z[:n] @ beta, "jac": lambda z: np.r_[-beta, e(3)]},
    ]
    w0 = np.full(n, 1 / n)
    s0 = portfolio_stats(w0, mu, beta, cov, rf)            # feasible start: deviations = shortfalls
    d0 = [max(0, T["Return"] - s0["Return"]), max(0, s0["Volatility"] - T["Volatility"]),
          max(0, T["Sharpe"] - s0["Sharpe"]), max(0, s0["Beta"] - T["Beta"])]
    bounds = [(g["min_weight"], g["max_weight"])] * n + [(0, None)] * 4
    res = minimize(obj, np.r_[w0, d0], jac=obj_jac, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    # SLSQP mode 8 ("positive directional derivative") = no further descent step at a point that
    # can already be optimal; accept it only if that point satisfies every constraint and bound.
    ok_at_stall = res.status == 8 and all(abs(c["fun"](res.x)) <= 1e-6 if c["type"] == "eq"
                                          else c["fun"](res.x) >= -1e-6 for c in cons) \
        and all(lo - 1e-9 <= x <= (hi if hi is not None else np.inf) + 1e-9 for x, (lo, hi) in zip(res.x, bounds))
    if not (res.success or ok_at_stall):
        raise RuntimeError(f"GP failed: {res.message}")
    w = np.clip(res.x[:n], 0, None)
    w /= w.sum()
    st = portfolio_stats(w, mu, beta, cov, rf)
    dev = {"Return": T["Return"] - st["Return"], "Volatility": st["Volatility"] - T["Volatility"],
           "Sharpe": T["Sharpe"] - st["Sharpe"], "Beta": st["Beta"] - T["Beta"]}
    info = {**{"GP_" + k: v for k, v in st.items()},
            **{"Dev_" + k: max(0.0, v) for k, v in dev.items()},
            "GoalsMet": sum(v <= 1e-6 for v in dev.values()), "GP_Status": res.message,
            "GP_Objective": res.fun}
    return w, info


def max_sharpe_weights(mu, cov, rf, lo, hi):
    """Classical Markowitz tangency portfolio (long-only, same weight bounds as the GP)."""
    n = len(mu)
    sig = lambda w: np.sqrt(w @ cov @ w)
    res = minimize(lambda w: -(w @ mu - rf) / sig(w), np.full(n, 1 / n),
                   jac=lambda w: -(mu / sig(w) - (w @ mu - rf) * (cov @ w) / sig(w) ** 3),
                   method="SLSQP", bounds=[(lo, hi)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"Markowitz failed: {res.message}")
    w = np.clip(res.x, 0, None)
    return w / w.sum()


if __name__ == "__main__":
    # Two uncorrelated stocks: A (ret 30%, vol 20%, beta 1.4), B (ret 10%, vol 10%, beta 0.6).
    mu, beta, cov = np.array([0.3, 0.1]), np.array([1.4, 0.6]), np.diag([0.04, 0.01])
    g = {"targets": {"Return": 0.2, "Volatility": 0.2, "Sharpe": 0.5, "Beta": 1.0},
         "priorities": dict.fromkeys(GOALS, 1), "min_weight": 0.0, "max_weight": 1.0}
    w, info = goal_weights(mu, beta, cov, g, 0.0)
    # Return >= 0.2 needs wA >= 0.5; Beta <= 1 needs wA <= 0.5 -> unique wA = 0.5, all goals met.
    assert np.allclose(w, [0.5, 0.5], atol=1e-4) and info["GoalsMet"] == 4, (w, info)
    # Unattainable return target: GP trades it off but keeps the weights valid.
    g["targets"]["Return"] = 0.5
    w, info = goal_weights(mu, beta, cov, g, 0.0)
    assert np.isclose(w.sum(), 1) and info["Dev_Return"] > 0, info
    wm = max_sharpe_weights(mu, cov, 0.0, 0, 1)
    # tangency with diagonal cov: w ~ mu / var -> (7.5, 10) normalised
    assert np.allclose(wm, np.array([7.5, 10]) / 17.5, atol=1e-3), wm
    print("ok")
