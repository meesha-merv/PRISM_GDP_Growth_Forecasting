"""Train the final PRISM model and write the deployment bundle.

    python train_final.py            # writes ../deploy/prism-space/bundle/

For each live cutoff (June and August 2026) and horizon (h = 0: 2026,
h = 1: 2027):
  1. rolling-origin out-of-sample predictions of the final model (arm C),
     2000-2024, which give the conformal interval and the backtest chart;
  2. a final fit on every country-year whose outcome is known;
  3. forecasts for all economies at origin 2026, with 80% intervals and
     per-forecast drivers (change in the forecast when a feature group is
     replaced by its training median).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from prism import config as C, evaluate as E, final as FN, live as L, models as M

OUT = C.MODEL_DIR.parent / "deploy" / "prism-space" / "bundle"
OUT.mkdir(parents=True, exist_ok=True)
REGION_MAP = None
INCOME_MAP = {"low_income": 0, "lower_middle_income": 1, "upper_middle_income": 2, "high_income": 3}
GROUP_LABELS = {"growth": "Growth history", "annual": "Annual structure (WDI)",
                "monthly": "Monthly indicators (GEM)", "quarterly": "Quarterly GDP"}


def interval_params(pred, model, window=10, min_hist=8):
    """Scaled split-conformal from the last `window` years of OOS residuals:
    returns the quantile q and each country's residual scale."""
    d = pred[pred["model"] == model].dropna(subset=["pred"]).copy()
    d["res"] = d["y"] - d["pred"]
    glob = float(np.median(np.abs(d["res"])))
    sc = d.groupby("iso")["res"].agg(lambda r: np.median(np.abs(r)) if len(r) >= min_hist else np.nan)
    sc = sc.clip(lower=0.25 * glob).fillna(glob)
    last = d["target_year"].max()
    cal = d[d["target_year"] > last - window]
    score = np.abs(cal["res"]) / cal["iso"].map(sc).fillna(glob)
    q = float(np.quantile(score, min(1, (1 - C.ALPHA) * (1 + 1 / len(score)))))
    return q, sc.to_dict(), glob


def main():
    global REGION_MAP
    meta = {"sklearn_version": sklearn.__version__, "backend": "LightGBM" if M.HAS_LGB else "HistGradientBoosting",
            "live_origin": L.LIVE_ORIGIN, "settings": []}
    forecasts, backtests, drivers = [], [], []
    for m in L.LIVE_CUTOFFS:
        for h in (0, 1):
            print(f"cutoff {m}, h {h}", flush=True)
            pred, _ = FN.rolling_arms(m, h, arms=("C",))
            name = pred["model"].iloc[0]
            q, scale, glob = interval_params(pred, name)
            sch = FN.tuned_schedule(4, h)
            params = FN._params_for(sch, 2024)

            live = L.build_live_store(m, h)
            fg = FN.groups(live)
            feats = FN.arm_features(live, "C")
            if REGION_MAP is None:
                REGION_MAP = {r: i for i, r in enumerate(sorted(live["world_6region"].dropna().unique()))}
            train = live[(live["target_year"] <= E.last_realised_year(L.LIVE_ORIGIN, m))
                         & (live["year"] >= E.TRAIN_FIRST_ORIGIN) & live["y"].notna()]
            mdl = FN.TunedGBM(feats, params, name="PRISM final").fit(train)
            test = live[live["year"] == L.LIVE_ORIGIN].copy()
            p = mdl.predict(test)
            sc = test["iso"].map(scale).fillna(glob).to_numpy()
            half = q * sc
            # drivers: replace each group by its training median
            med = train[mdl.features].median(numeric_only=True)
            drv = {}
            for g, cols in fg.items():
                if g not in GROUP_LABELS:
                    continue
                cols = [c for c in cols if c in mdl.features]
                if not cols:
                    continue
                tp = test.copy()
                tp[cols] = med[cols].to_numpy()
                d = p - mdl.predict(tp)
                # a block the economy does not report has no effect to explain
                d[test[cols].isna().all(axis=1).to_numpy()] = np.nan
                drv[g] = d
            tag = f"m{m:02d}_h{h}"
            joblib.dump({"estimator": mdl.m_, "features": mdl.features, "params": params,
                         "region_map": REGION_MAP, "income_map": INCOME_MAP,
                         "q": q, "scale": scale, "global_scale": glob,
                         "train_median": med.to_dict(), "cutoff_month": m, "horizon": h,
                         "n_train": int(len(train))}, OUT / f"model_{tag}.joblib", compress=3)
            fc = pd.DataFrame({"iso": test["iso"].to_numpy(), "name": test["name"].to_numpy(),
                               "region": test["world_6region"].to_numpy(), "is_sa": test["is_sa"].to_numpy(),
                               "cutoff_month": m, "horizon": h, "target_year": test["target_year"].to_numpy(),
                               "forecast": p, "lo80": p - half, "hi80": p + half,
                               "gep_june2026": test["gep_2026_forecast"].to_numpy(),
                               "last_growth_used": test["g_lag1_rt"].to_numpy(),
                               "months_ahead": test["months_ahead"].to_numpy()})
            for g, v in drv.items():
                fc["drv_" + g] = v
            forecasts.append(fc.dropna(subset=["forecast"]))
            # inputs the app re-scores live
            keep = ["iso", "year", "target_year", "income_groups", "world_6region", "g_lag1_rt"] + mdl.features
            test[list(dict.fromkeys(keep))].to_csv(OUT / f"inputs_{tag}.csv", index=False)
            bt = pred[["iso", "origin", "target_year", "y", "pred", "is_sa"]].copy()
            bt["cutoff_month"], bt["horizon"] = m, h
            backtests.append(bt)
            # benchmark history for the chart
            meta["settings"].append({"tag": tag, "cutoff_month": m, "horizon": h, "q": q,
                                     "params": params, "n_train": int(len(train)),
                                     "n_features": len(mdl.features)})
    fc = pd.concat(forecasts, ignore_index=True)
    fc.round(3).to_csv(OUT / "forecasts_2026.csv", index=False)
    bt = pd.concat(backtests, ignore_index=True)
    # IMF WEO and naive from the ladder (April/June cutoffs) for the chart
    lad = []
    for m, h in ((4, 0), (4, 1), (6, 0), (6, 1)):
        pr = pd.read_pickle(C.OUT_DIR / "cache" / f"pred_m{m:02d}_h{h}.pkl")
        pr = pr[pr["model"].isin(["IMF WEO", "Naive (real-time)"])][["model", "iso", "target_year", "pred"]]
        pr["cutoff_month"], pr["horizon"] = m, h
        lad.append(pr)
    lad = pd.concat(lad).pivot_table(index=["iso", "target_year", "cutoff_month", "horizon"],
                                     columns="model", values="pred").reset_index()
    lad.columns = ["iso", "target_year", "cutoff_month", "horizon", "weo", "naive"]
    bt.round(3).to_csv(OUT / "backtest.csv", index=False)
    lad.round(3).to_csv(OUT / "benchmarks.csv", index=False)
    with open(OUT / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(fc[fc.is_sa == 1][["iso", "cutoff_month", "horizon", "target_year", "forecast", "lo80", "hi80",
                             "gep_june2026"]].round(2).to_string())


if __name__ == "__main__":
    main()
