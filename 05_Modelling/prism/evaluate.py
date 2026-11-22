"""Rolling-origin evaluation, scores, paired tests and conformal intervals."""
import copy
import time

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from . import features as F

TRAIN_FIRST_ORIGIN = 1992   # monthly GEM data start in 1990; the EDA model window


def load_store(cutoff_month, horizon, cache=True):
    path = C.OUT_DIR / "cache" / f"store_m{cutoff_month:02d}_h{horizon}.pkl"
    path.parent.mkdir(exist_ok=True)
    if cache and path.exists():
        return pd.read_pickle(path)
    s = F.build_store(cutoff_month, horizon)
    s.to_pickle(path)
    return s


def last_realised_year(t, m):
    """Latest target year whose outcome is (preliminarily) known at the cutoff."""
    return t - 1 if m >= C.GDP_PRELIM_MONTH else t - 2


def rolling_origin(store, models, test_years=C.TEST_YEARS, verbose=False):
    """Expanding-window evaluation. At each origin t every model is refit on
    rows whose outcome was known at the cutoff, then predicts year t for
    every country. Returns long-form predictions."""
    m = int(store["cutoff_month"].iloc[0])
    out = []
    for t in test_years:
        train = store[(store["target_year"] <= last_realised_year(t, m))
                      & (store["year"] >= TRAIN_FIRST_ORIGIN) & store["y"].notna()]
        test = store[(store["year"] == t) & store["y"].notna()]
        if test.empty:
            continue
        for mdl in models:
            t0 = time.time()
            model = copy.deepcopy(mdl).fit(train)
            pred = model.predict(test)
            out.append(pd.DataFrame({
                "model": mdl.name, "iso": test["iso"].to_numpy(), "origin": t,
                "target_year": test["target_year"].to_numpy(), "y": test["y"].to_numpy(),
                "pred": pred, "is_sa": test["is_sa"].to_numpy(),
                "months_ahead": test["months_ahead"].to_numpy(),
                "cutoff_month": m, "horizon": int(store["horizon"].iloc[0]),
                "n_train": len(train)}))
            if verbose:
                print(f"  origin {t} {mdl.name:28s} {time.time()-t0:5.1f}s")
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ scores
def scores(df, ref="Naive (real-time)"):
    """RMSE, MAE, MAPE, R2 and skill against a reference, on rows where
    every model in df has a prediction (a common sample)."""
    wide = df.pivot_table(index=["iso", "target_year"], columns="model", values="pred")
    y = df.drop_duplicates(["iso", "target_year"]).set_index(["iso", "target_year"])["y"]
    wide = wide.dropna()
    y = y.reindex(wide.index)
    rows = []
    for mdl in wide.columns:
        e = wide[mdl] - y
        nz = y.abs() > 1.0          # MAPE explodes near zero growth; see report
        rows.append({
            "model": mdl, "n": len(e),
            "RMSE": np.sqrt(np.mean(e ** 2)),
            "MAE": np.mean(np.abs(e)),
            "MAPE_%": np.mean(np.abs(e[nz] / y[nz])) * 100,
            "R2": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2),
        })
    res = pd.DataFrame(rows).set_index("model")
    if ref in res.index:
        res["RMSE_vs_naive"] = res["RMSE"] / res.loc[ref, "RMSE"]
    return res.sort_values("RMSE")


def score_table(pred, models=None, sa_only=True, years=None, exclude_crisis=False, ref="Naive (real-time)"):
    d = pred if models is None else pred[pred["model"].isin(models)]
    if sa_only:
        d = d[d["is_sa"] == 1]
    if years is not None:
        d = d[d["target_year"].between(*years)]
    if exclude_crisis:
        d = d[~d["target_year"].isin(C.CRISIS_YEARS)]
    return scores(d, ref=ref)


def dm_test(pred, model, bench="IMF WEO", h=0, sa_only=True):
    """Diebold-Mariano test (squared loss) with the Harvey-Leybourne-Newbold
    small-sample correction, on the pooled South Asian country-years.
    Negative mean_loss_diff = `model` beats `bench`."""
    d = pred[pred["model"].isin([model, bench])]
    if sa_only:
        d = d[d["is_sa"] == 1]
    w = d.pivot_table(index=["target_year", "iso"], columns="model", values="pred").dropna()
    y = d.drop_duplicates(["iso", "target_year"]).set_index(["target_year", "iso"])["y"].reindex(w.index)
    dl = (w[model] - y) ** 2 - (w[bench] - y) ** 2
    n = len(dl)
    dbar = dl.mean()
    # Newey-West variance with h lags (forecast errors overlap when h > 0)
    gamma0 = np.var(dl, ddof=0)
    var = gamma0
    for k in range(1, h + 1):
        cov = np.mean((dl[k:].to_numpy() - dbar) * (dl[:-k].to_numpy() - dbar))
        var += 2 * (1 - k / (h + 1)) * cov
    dm = dbar / np.sqrt(var / n)
    hln = dm * np.sqrt((n + 1 - 2 * (h + 1) + (h + 1) * h / n) / n)
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return {"model": model, "bench": bench, "n": n, "mean_loss_diff": dbar, "DM_HLN": hln, "p_value": p}


# ------------------------------------------------------------------ conformal intervals
def conformal(pred, model, alpha=C.ALPHA, window=10, method="global", gamma=0.05, min_hist=8):
    """Split-conformal 80% intervals built ONLY from earlier out-of-sample
    residuals (origins whose outcome was known at the time).

    method = "global"   : one half-width for every country (proposal design)
             "scaled"   : residuals divided by each country's historical
                          residual scale, so volatile economies get wider bands
             "adaptive" : "scaled" plus adaptive conformal inference
                          (Gibbs & Candes 2021): each country's miscoverage
                          level is nudged after every miss or hit.
    """
    d = pred[pred["model"] == model].dropna(subset=["pred"]).copy()
    d["res"] = d["y"] - d["pred"]
    h = int(d["horizon"].iloc[0])
    m = int(d["cutoff_month"].iloc[0])
    rows = []
    alpha_c = {}
    pending = []    # (target_year, iso, hit): ACI updates wait until the outcome is public
    for t in sorted(d["origin"].unique()):
        known = last_realised_year(t, m)
        if method == "adaptive":
            for ty, iso, ht in [p for p in pending if p[0] <= known]:
                alpha_c[iso] = alpha_c.get(iso, alpha) + gamma * (alpha - (0 if ht else 1))
            pending = [p for p in pending if p[0] > known]
        cal = d[(d["target_year"] <= known) & (d["origin"] >= t - window - h)]
        hist = d[d["target_year"] <= known]
        test = d[d["origin"] == t]
        if len(cal) < 50:
            continue
        if method == "global":
            q = np.quantile(np.abs(cal["res"]), min(1, (1 - alpha) * (1 + 1 / len(cal))))
            half = np.full(len(test), q)
        else:
            # country residual scale: robust spread of that country's past
            # OOS residuals; fall back to the global spread when history is short
            glob = np.median(np.abs(hist["res"]))
            sc = hist.groupby("iso")["res"].agg(lambda r: np.median(np.abs(r)) if len(r) >= min_hist else np.nan)
            sc = sc.clip(lower=0.25 * glob)
            cal_s = cal["iso"].map(sc).fillna(glob)
            score = np.abs(cal["res"]) / cal_s
            scale = test["iso"].map(sc).fillna(glob).to_numpy()
            if method == "scaled":
                q = np.quantile(score, min(1, (1 - alpha) * (1 + 1 / len(score))))
                half = q * scale
            else:  # adaptive, per country
                half = []
                for iso, s_ in zip(test["iso"], scale):
                    a = alpha_c.get(iso, alpha)
                    a_eff = float(np.clip(a, 0.005, 0.995))
                    q = np.quantile(score, min(1, (1 - a_eff) * (1 + 1 / len(score))))
                    half.append(q * s_)
                half = np.array(half)
        lo, hi = test["pred"].to_numpy() - half, test["pred"].to_numpy() + half
        hit = (test["y"].to_numpy() >= lo) & (test["y"].to_numpy() <= hi)
        if method == "adaptive":
            pending += list(zip(test["target_year"], test["iso"], hit))
        rows.append(pd.DataFrame({"iso": test["iso"].to_numpy(), "origin": t,
                                  "target_year": test["target_year"].to_numpy(),
                                  "y": test["y"].to_numpy(), "pred": test["pred"].to_numpy(),
                                  "lo": lo, "hi": hi, "covered": hit,
                                  "is_sa": test["is_sa"].to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    out["width"] = out["hi"] - out["lo"]
    out["method"] = method
    out["model"] = model
    return out


def coverage_table(iv):
    rows = []
    for (mdl, meth), g in iv.groupby(["model", "method"]):
        for label, sub in [("South Asia, all years", g[g.is_sa == 1]),
                           ("South Asia, excl. 2020-21", g[(g.is_sa == 1) & ~g.target_year.isin(C.CRISIS_YEARS)]),
                           ("South Asia, 2020-21 only", g[(g.is_sa == 1) & g.target_year.isin(C.CRISIS_YEARS)]),
                           ("Rest of world", g[g.is_sa == 0])]:
            rows.append({"model": mdl, "method": meth, "sample": label, "n": len(sub),
                         "coverage_%": sub["covered"].mean() * 100, "mean_width_pp": sub["width"].mean()})
    return pd.DataFrame(rows)
