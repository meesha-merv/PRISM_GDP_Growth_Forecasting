"""Final-stage experiments (weeks 5-8 of the plan).

  * Quarterly GDP block (latest quarter y/y, four-quarter growth, and growth
    of the target reporting year's published quarters).
  * Final feature set = growth history + annual WDI + monthly GEM + quarterly
    GDP. Pink Sheet indices are dropped: the ablation showed they add nothing.
  * GBM hyperparameters re-tuned every five origins on inner time splits
    (the three origins just before the block, all economies), never on the
    test years.
  * RQ1 arms: annual-only revised (A), annual-only real-time (B),
    mixed-frequency real-time (C, the final model).
  * Permutation importance of the final model, by feature and by group.
  * RQ2: a crisis-aware conformal interval whose width is driven by a model
    of the absolute error learned on the global panel from real-time stress
    signals, scored by coverage and interval score.
"""
import copy
import itertools
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config as C
from . import evaluate as E
from . import features as F
from . import models as M

QGDP_LAG = 3      # months after the end of a quarter before its GDP is public (conservative)


# ------------------------------------------------------------------ quarterly GDP block
def _load_qgdp():
    q = pd.read_csv(C.RAW_DIR / "gem_quarterly_gdp.csv")
    q = q[q["indicator"] == "NYGDPMKTPSAKD"]
    q["period"] = pd.PeriodIndex(q["date"].str.replace("Q", "-Q"), freq="Q")
    s = q.set_index(["iso", "period"])["value"].sort_index()
    return s[~s.index.duplicated()]


def quarterly_block(origins, m, horizon):
    s = _load_qgdp()
    out = {"q_yoy_last": [], "q_yoy_4q": [], "q_fytd": []}
    by_iso = {iso: g.droplevel(0) for iso, g in s.groupby(level="iso")}
    for iso, t in zip(origins["iso"], origins["year"]):
        t = int(t)
        ser = by_iso.get(iso)
        if ser is None:
            for k in out:
                out[k].append(np.nan)
            continue
        cut = pd.Period(year=t, month=m, freq="M") - QGDP_LAG
        lastq = cut.asfreq("Q")
        if cut.month % 3 != 0:      # quarter not yet complete at cut
            lastq = lastq - 1
        def g(p):
            return ser.get(p, np.nan)
        cur, prev = g(lastq), g(lastq - 4)
        out["q_yoy_last"].append((cur / prev - 1) * 100 if prev == prev and cur == cur else np.nan)
        c4 = [g(lastq - i) for i in range(4)]
        p4 = [g(lastq - 4 - i) for i in range(4)]
        out["q_yoy_4q"].append((np.sum(c4) / np.sum(p4) - 1) * 100
                               if not np.isnan(c4 + p4).any() else np.nan)
        val = np.nan
        if horizon == 0:
            (sy, sm), _ = C.fy_window(iso, t)
            first_q = pd.Period(year=sy, month=sm, freq="M").asfreq("Q")
            if pd.Period(year=sy, month=sm, freq="M").month % 3 != 1:
                first_q = first_q + 1          # start at the first full quarter
            qs = [p for p in pd.period_range(first_q, lastq, freq="Q")] if lastq >= first_q else []
            if qs:
                c = [g(p) for p in qs]
                pv = [g(p - 4) for p in qs]
                if not np.isnan(c + pv).any():
                    val = (np.sum(c) / np.sum(pv) - 1) * 100
        out["q_fytd"].append(val)
    return pd.DataFrame(out)


def load_final_store(m, h):
    path = C.OUT_DIR / "cache" / f"store_final_m{m:02d}_h{h}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    s = E.load_store(m, h)
    q = quarterly_block(s[["iso", "year"]], m, h)
    s = pd.concat([s.reset_index(drop=True), q], axis=1)
    s.to_pickle(path)
    return s


def build_store_live(m, h, last_origin=2026):
    """Store including the live origin (2026), for the deployed model."""
    s = F.build_store(m, h, last_origin=last_origin)
    q = quarterly_block(s[["iso", "year"]], m, h)
    return pd.concat([s.reset_index(drop=True), q], axis=1)


def groups(store):
    fg = F.feature_groups(store)
    fg["quarterly"] = [c for c in store.columns if c.startswith("q_") and c[2:5] in ("yoy", "fyt")]
    fg["monthly"] = [c for c in fg["monthly"] if not c.startswith("q_")]
    return fg


def arm_features(store, arm):
    fg = groups(store)
    if arm == "A":
        return fg["growth_revised"] + fg["annual"]
    if arm == "B":
        return fg["growth"] + fg["annual"]
    if arm == "C0":                     # mixed frequency, monthly only
        return fg["growth"] + fg["annual"] + fg["monthly"]
    if arm == "C":                      # final model: + quarterly GDP
        return fg["growth"] + fg["annual"] + fg["monthly"] + fg["quarterly"]
    raise ValueError(arm)


# ------------------------------------------------------------------ tuned GBM
GRID = [dict(max_leaf_nodes=l, min_samples_leaf=s, learning_rate=lr, max_iter=it)
        for l, s, (lr, it) in itertools.product((7, 15, 31), (20, 60),
                                                ((0.03, 400), (0.05, 800), (0.10, 200)))]
DEFAULT = dict(max_leaf_nodes=15, min_samples_leaf=40, learning_rate=0.03, max_iter=400)
GRID.append(dict(DEFAULT))          # the untuned preview, so tuning can only match or beat it


class TunedGBM(M.GBM):
    """Histogram GBM with given hyperparameters; the row filter (needs a
    real-time last-year value) matches the ladder GBM. Training rows are
    weighted so that the guide variable is the raw (clipped) target."""
    def __init__(self, features, params=None, name="GBM tuned", seed=C.RANDOM_STATE, require_rt=True):
        super().__init__(features, name=name, seed=seed)
        self.params = dict(DEFAULT if params is None else params)
        self.require_rt = require_rt
        self.lag_col = "g_lag1_rt"

    def fit(self, train):
        tr = train.dropna(subset=[self.lag_col])
        y = M._clip_target(tr["y"].to_numpy())
        # a feature with fewer than two distinct training values carries no
        # information (e.g. fiscal-year-to-date quarterly growth at an April
        # cutoff); drop it for this fold
        self.features = [f for f in self.features if tr[f].nunique(dropna=True) >= 2]
        X = self._X(tr)
        cat_mask = [c == "region_code" for c in X.columns]
        self.m_ = HistGradientBoostingRegressor(l2_regularization=1.0, categorical_features=cat_mask,
                                                random_state=self.seed, **self.params)
        self.m_.fit(X, y)
        return self

    def predict(self, test):
        out = self.m_.predict(self._X(test))
        out[test[self.lag_col].isna().to_numpy()] = np.nan
        return out


def _train_rows(store, t, m):
    return store[(store["target_year"] <= E.last_realised_year(t, m))
                 & (store["year"] >= E.TRAIN_FIRST_ORIGIN) & store["y"].notna()]


def tune(store, feats, block_start, n_inner=3, verbose=False):
    """Pick hyperparameters on the n_inner origins before block_start.
    Score = RMSE on all economies with the target clipped to the training
    1st/99th percentiles (so a single war year does not choose the model)."""
    m = int(store["cutoff_month"].iloc[0])
    res = []
    for p in GRID:
        se, n = 0.0, 0
        for t in range(block_start - n_inner, block_start):
            tr = _train_rows(store, t, m)
            te = store[(store["year"] == t) & store["y"].notna()]
            if te.empty or len(tr) < 500:
                continue
            mdl = TunedGBM(feats, p).fit(tr)
            pr = mdl.predict(te)
            lo, hi = np.nanpercentile(tr["y"], [1, 99])
            e = (pr - np.clip(te["y"].to_numpy(), lo, hi))
            e = e[~np.isnan(e)]
            se += np.sum(e ** 2)
            n += len(e)
        res.append({**p, "rmse": np.sqrt(se / max(n, 1))})
    res = pd.DataFrame(res).sort_values("rmse")
    if verbose:
        print(res.head(3))
    best = res.iloc[0][["max_leaf_nodes", "min_samples_leaf", "learning_rate", "max_iter"]].to_dict()
    best = {k: (float(v) if k == "learning_rate" else int(v)) for k, v in best.items()}
    return best, res


BLOCKS = [2000, 2005, 2010, 2015, 2020]


def tuned_schedule(m, h, arm="C", verbose=False):
    path = C.OUT_DIR / "cache" / f"tune_m{m:02d}_h{h}_{arm}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    store = load_final_store(m, h)
    feats = arm_features(store, arm)
    rows = []
    for b in BLOCKS:
        best, res = tune(store, feats, b, verbose=verbose)
        rows.append({"block_start": b, **best, "inner_rmse": res["rmse"].iloc[0],
                     "default_inner_rmse": res[(res.max_leaf_nodes == 15) & (res.min_samples_leaf == 40)
                                               & (res.learning_rate == 0.03) & (res.max_iter == 400)]["rmse"].iloc[0]})
    sch = pd.DataFrame(rows)
    sch.to_pickle(path)
    return sch


def _params_for(sch, t):
    r = sch[sch["block_start"] <= t].iloc[-1]
    return {"max_leaf_nodes": int(r.max_leaf_nodes), "min_samples_leaf": int(r.min_samples_leaf),
            "learning_rate": float(r.learning_rate), "max_iter": int(r.max_iter)}


def rolling_arms(m, h, arms=("A", "B", "C0", "C"), sch=None, importance=False, n_perm=5, verbose=False):
    """Rolling-origin predictions for the RQ1 arms with tuned GBMs.
    If importance=True, also permutation importance for arm C."""
    path = C.OUT_DIR / "cache" / f"arms_m{m:02d}_h{h}_{''.join(arms)}.pkl"
    ipath = C.OUT_DIR / "cache" / f"perm_m{m:02d}_h{h}.pkl"
    if path.exists() and (not importance or ipath.exists()):
        return pd.read_pickle(path), (pd.read_pickle(ipath) if importance else None)
    store = load_final_store(m, h)
    # hyperparameters are tuned at the April cutoff for each horizon and
    # reused at the other cutoffs (tuning every cutoff would cost 12x)
    sch = tuned_schedule(4, h) if sch is None else sch
    rng = np.random.default_rng(0)
    test_years = [t for t in C.TEST_YEARS if t + h <= 2024]
    out, imp = [], []
    for t in test_years:
        tr = _train_rows(store, t, m)
        te = store[(store["year"] == t) & store["y"].notna()]
        p = _params_for(sch, t)
        for arm in arms:
            feats = arm_features(store, arm)
            name = {"A": "A. Annual-only, revised", "B": "B. Annual-only, real-time",
                    "C0": "C0. Mixed (monthly), real-time", "C": "C. Final: mixed (monthly+quarterly)"}[arm]
            mdl = TunedGBM(feats, p, name=name)
            if arm == "A":
                mdl.lag_col = "g_lag1_rt"   # same rows as other arms
            mdl.fit(tr)
            pred = mdl.predict(te)
            out.append(pd.DataFrame({"model": name, "iso": te["iso"].to_numpy(), "origin": t,
                                     "target_year": te["target_year"].to_numpy(), "y": te["y"].to_numpy(),
                                     "pred": pred, "is_sa": te["is_sa"].to_numpy(),
                                     "months_ahead": te["months_ahead"].to_numpy(),
                                     "cutoff_month": m, "horizon": h, "n_train": len(tr)}))
            if importance and arm == "C":
                y = te["y"].to_numpy()
                base = mdl.predict(te)
                ok = ~np.isnan(base)
                sa = (te["is_sa"].to_numpy() == 1) & ok
                for f in feats + ["__group__" + g for g in ("growth", "annual", "monthly", "quarterly")]:
                    if f.startswith("__group__"):
                        cols = [c for c in groups(store)[f[9:]] if c in feats]
                    else:
                        cols = [f]
                    for r in range(n_perm):
                        tp = te.copy()
                        idx = rng.permutation(len(tp))
                        tp[cols] = tp[cols].to_numpy()[idx]
                        pp = mdl.predict(tp)
                        imp.append({"feature": f, "origin": t, "rep": r,
                                    "d_se_all": np.mean((pp[ok] - y[ok]) ** 2) - np.mean((base[ok] - y[ok]) ** 2),
                                    "d_se_sa": np.mean((pp[sa] - y[sa]) ** 2) - np.mean((base[sa] - y[sa]) ** 2)})
        if verbose:
            print(f"  m{m} h{h} origin {t} done")
    pred = pd.concat(out, ignore_index=True)
    pred.to_pickle(path)
    if importance:
        imp = pd.DataFrame(imp)
        imp.to_pickle(ipath)
        return pred, imp
    return pred, None


def importance_table(imp):
    g = imp.groupby(["feature", "origin"])[["d_se_all", "d_se_sa"]].mean().groupby("feature").mean()
    g["rank_all"] = g["d_se_all"].rank(ascending=False)
    return g.sort_values("d_se_all", ascending=False)


# ------------------------------------------------------------------ DM tests (squared and absolute loss)
def dm_any(pred, model, bench, h=0, loss="se", years=None):
    d = pred[pred["model"].isin([model, bench]) & (pred["is_sa"] == 1)]
    if years is not None:
        d = d[d["target_year"].between(*years)]
    w = d.pivot_table(index=["target_year", "iso"], columns="model", values="pred").dropna()
    y = d.drop_duplicates(["iso", "target_year"]).set_index(["target_year", "iso"])["y"].reindex(w.index)
    f = (lambda e: e ** 2) if loss == "se" else np.abs
    dl = f(w[model] - y) - f(w[bench] - y)
    n = len(dl)
    dbar = dl.mean()
    var = np.var(dl, ddof=0)
    for k in range(1, h + 1):
        var += 2 * (1 - k / (h + 1)) * np.mean((dl[k:].to_numpy() - dbar) * (dl[:-k].to_numpy() - dbar))
    dm = dbar / np.sqrt(var / n)
    hln = dm * np.sqrt((n + 1 - 2 * (h + 1) + (h + 1) * h / n) / n)
    from scipy import stats
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return {"model": model, "bench": bench, "loss": loss, "n": n, "mean_loss_diff": dbar, "DM_HLN": hln, "p_value": p}


# ------------------------------------------------------------------ RQ2: crisis-aware intervals
STRESS = ["exports_m_yoy3", "imports_m_yoy3", "ip_yoy3", "reserves_m_yoy3", "fx_m_yoy3",
          "exports_m_fytd", "ip_fytd", "q_yoy_last", "g_lag1_rt", "g_lag2_rt"]


def stress_frame(store):
    """Real-time stress signals: country-level monthly swings plus the
    cross-country (global) median of the same signals at the cutoff, and
    the size of the last growth swing."""
    s = store[["iso", "year"] + [c for c in STRESS if c in store.columns]].copy()
    for c in ["exports_m_yoy3", "imports_m_yoy3", "ip_yoy3", "fx_m_yoy3", "q_yoy_last"]:
        if c in s:
            s["glob_" + c] = s.groupby("year")[c].transform("median")
            s["dev_" + c] = s[c] - s["glob_" + c]
    s["swing"] = (s["g_lag1_rt"] - s["g_lag2_rt"]).abs()
    return s.drop(columns=["g_lag1_rt", "g_lag2_rt"])


def crisis_conformal(pred, store, model, alpha=C.ALPHA, window=10, min_hist=8):
    """Normalised split-conformal interval. Half-width = q * sigma_hat, where
    sigma_hat is (a) the country's past residual scale times (b) a stress
    multiplier from a GBM fitted on past absolute residuals of ALL economies
    against real-time stress signals. Everything uses only residuals whose
    outcome was public at the origin. q is the conformal quantile of the
    normalised scores in the calibration window."""
    d = pred[pred["model"] == model].dropna(subset=["pred"]).copy()
    d["res"] = d["y"] - d["pred"]
    h = int(d["horizon"].iloc[0]); m = int(d["cutoff_month"].iloc[0])
    st = stress_frame(store)
    d = d.merge(st, left_on=["iso", "origin"], right_on=["iso", "year"], how="left").drop(columns="year")
    scols = [c for c in st.columns if c not in ("iso", "year")]
    rows = []
    for t in sorted(d["origin"].unique()):
        known = E.last_realised_year(t, m)
        hist = d[d["target_year"] <= known]
        cal = hist[hist["origin"] >= t - window - h]
        test = d[d["origin"] == t]
        if len(cal) < 50 or len(hist) < 300:
            continue
        glob = np.median(np.abs(hist["res"]))
        sc = hist.groupby("iso")["res"].agg(lambda r: np.median(np.abs(r)) if len(r) >= min_hist else np.nan)
        sc = sc.clip(lower=0.25 * glob)
        base_h = hist["iso"].map(sc).fillna(glob).to_numpy()
        # stress multiplier: GBM on log(|res| / country scale), fitted on the
        # global panel. Calibration scores are cross-fitted (each calibration
        # year is scored by a model that did not see that year), otherwise the
        # in-sample fit makes the scores too small and the band too narrow.
        target = np.log((np.abs(hist["res"].to_numpy()) + 0.1) / base_h)
        def fit(rows):
            mdl = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
                                                min_samples_leaf=50, random_state=0)
            return mdl.fit(hist.loc[rows, scols], target[hist.index.get_indexer(rows)])
        hist = hist.copy()
        full = fit(hist.index)
        centre = np.median(target)
        def sig(df, mdl):
            b = df["iso"].map(sc).fillna(glob).to_numpy()
            return b * np.exp(mdl.predict(df[scols]) - centre)
        s_cal = np.empty(len(cal))
        years = np.sort(cal["target_year"].unique())
        for chunk in np.array_split(years, min(5, len(years))):
            out_rows = hist.index[~hist["target_year"].isin(chunk)]
            mdl_k = fit(out_rows)
            mask = cal["target_year"].isin(chunk).to_numpy()
            s_cal[mask] = sig(cal[mask], mdl_k)
        score = np.abs(cal["res"].to_numpy()) / s_cal
        q = np.quantile(score, min(1, (1 - alpha) * (1 + 1 / len(score))))
        half = q * sig(test, full)
        lo, hi = test["pred"].to_numpy() - half, test["pred"].to_numpy() + half
        y = test["y"].to_numpy()
        rows.append(pd.DataFrame({"iso": test["iso"].to_numpy(), "origin": t,
                                  "target_year": test["target_year"].to_numpy(), "y": y,
                                  "pred": test["pred"].to_numpy(), "lo": lo, "hi": hi,
                                  "covered": (y >= lo) & (y <= hi), "is_sa": test["is_sa"].to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    out["width"] = out["hi"] - out["lo"]
    out["method"] = "crisis-aware"
    out["model"] = model
    return out


def interval_score(iv, alpha=C.ALPHA):
    """Winkler / interval score: width + (2/alpha) * distance outside. Lower is better."""
    lo, hi, y = iv["lo"], iv["hi"], iv["y"]
    return (hi - lo) + (2 / alpha) * ((lo - y).clip(lower=0) + (y - hi).clip(lower=0))


def rq2_table(ivs):
    iv = pd.concat(ivs, ignore_index=True)
    iv["IS"] = interval_score(iv)
    rows = []
    crises = {"Sri Lanka 2022": (iv.iso == "LKA") & (iv.target_year == 2022),
              "Afghanistan 2021": (iv.iso == "AFG") & (iv.target_year == 2021)}
    for meth, g in iv.groupby("method"):
        samples = [("South Asia, all years", g.is_sa == 1),
                   ("South Asia, excl. 2020-21", (g.is_sa == 1) & ~g.target_year.isin(C.CRISIS_YEARS)),
                   ("South Asia, 2020-21", (g.is_sa == 1) & g.target_year.isin(C.CRISIS_YEARS)),
                   ("Global, 2009", g.target_year == 2009),
                   ("Global, 2020", g.target_year == 2020),
                   ("Rest of world, all years", g.is_sa == 0)]
        for label, mask in samples:
            sub = g[mask]
            n = len(sub)
            cov = sub["covered"].mean()
            se = np.sqrt(cov * (1 - cov) / n) if n else np.nan
            rows.append({"method": meth, "sample": label, "n": n, "coverage_%": cov * 100,
                         "cov_lo95": max(0, cov - 1.96 * se) * 100, "cov_hi95": min(1, cov + 1.96 * se) * 100,
                         "mean_width_pp": sub["width"].mean(), "interval_score": sub["IS"].mean()})
        for label, mask in crises.items():
            sub = g[mask.loc[g.index]]
            if len(sub):
                rows.append({"method": meth, "sample": label, "n": len(sub),
                             "coverage_%": sub["covered"].mean() * 100,
                             "mean_width_pp": sub["width"].mean(), "interval_score": sub["IS"].mean(),
                             "lo": sub["lo"].iloc[0], "hi": sub["hi"].iloc[0], "y": sub["y"].iloc[0]})
    return pd.DataFrame(rows)
