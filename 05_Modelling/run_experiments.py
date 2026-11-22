"""Run every experiment in the progress report and save tables and figures.

    python run_experiments.py            # uses cached feature stores if present
    python run_experiments.py --fresh    # rebuild everything from the raw files

Outputs land in outputs/tables (CSV) and outputs/figures (PNG).
"""
import shutil
import sys
import time

import pandas as pd

from prism import config as C, experiments as X, models as M, reconcile as R

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def main():
    if "--fresh" in sys.argv:
        shutil.rmtree(C.OUT_DIR / "cache", ignore_errors=True)
    t0 = time.time()
    gbm = "LightGBM" if M.HAS_LGB else "GBM (HistGB)"
    print("Gradient boosting backend:", gbm)

    print("\n[1] Reporting-period reconciliation")
    X._save(R.label_alignment().round(3), "reconcile_weo_vs_wdi")
    X._save(R.gep_alignment().round(3), "reconcile_gep_vs_wdi")
    X._save(R.effective_horizons(4), "effective_horizon_april")
    print(R.label_alignment()[["iso", "MAE_shift_-1", "MAE_shift_+0", "MAE_shift_+1", "best_shift"]].round(2))

    print("\n[2] Baseline ladder")
    preds = {}
    for m, h in [(4, 0), (4, 1), (6, 0), (6, 1)]:
        tag = f"m{m:02d}_h{h}"
        preds[tag] = X.run_ladder(m, h)
        tabs = X.ladder_tables(preds[tag], tag)
        X.dm_table(preds[tag], h, tag)
        print(f"\n--- cutoff month {m}, horizon {h}: South Asia 2000-2022")
        print(tabs["sa_weo_window"].round(2))
    X.per_country(preds["m04_h0"])
    X.by_effective_horizon(preds["m04_h0"])
    X.truth_robustness(preds["m04_h0"])

    print("\n[3] Feature ablation (April, h=0)")
    print(X.ablation(4, 0))

    print("\n[4] Conformal intervals")
    for tag in ("m04_h0", "m04_h1"):
        iv, cov = X.interval_experiment(preds[tag], [gbm, "AR(2) pooled"], tag)
        print(tag); print(cov.round(1))

    print("\n[5] Monthly refresh (h=0, cutoffs Jan-Dec)")
    _, refresh = X.monthly_refresh()
    print(refresh)

    print("\n[6] Figures")
    t = "m04_h0"
    all_ = pd.read_csv(C.TAB_DIR / f"{t}_sa_weo_window.csv", index_col=0)
    ex = pd.read_csv(C.TAB_DIR / f"{t}_sa_weo_window_excl_crisis.csv", index_col=0)
    X.fig_ladder(all_, ex)
    a1 = pd.read_csv(C.TAB_DIR / "m04_h1_sa_weo_window.csv", index_col=0)
    e1 = pd.read_csv(C.TAB_DIR / "m04_h1_sa_weo_window_excl_crisis.csv", index_col=0)
    X.fig_ladder(a1, e1, "fig1b_baseline_ladder_h1.png",
                 "South Asia, April cutoff, next year (h = 1), 2001-2022")
    X.fig_refresh(refresh, gbm)
    iv = pd.read_pickle(C.OUT_DIR / "cache" / "intervals_m04_h0.pkl")
    X.fig_country_paths(preds[t][preds[t]["target_year"] >= 2000],
                        iv[(iv["model"] == gbm) & (iv["method"] == "scaled")], gbm)
    cov = pd.read_csv(C.TAB_DIR / "m04_h0_interval_coverage.csv", index_col=0)
    X.fig_coverage(cov, model=gbm)
    X.fig_horizon(pd.read_csv(C.TAB_DIR / "m04_h0_rmse_by_effective_horizon.csv", index_col=0), gbm)
    print(f"\nDone in {time.time() - t0:.0f}s. Tables: {C.TAB_DIR}  Figures: {C.FIG_DIR}")


if __name__ == "__main__":
    main()
