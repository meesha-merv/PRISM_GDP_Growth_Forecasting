"""Final experiments: tuning, RQ1 arms, importance, RQ2 intervals,
monthly refresh of the final model. Run after run_experiments.py.

    python run_final.py

Writes outputs/tables/final_*.csv and outputs/figures/final_*.png.
"""
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism import config as C, evaluate as E, experiments as X, final as FN

BLUE, ORANGE, AQUA, GREY, INK, NAVY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#2b2b29", "#16212B"
FINAL = "C. Final: mixed (monthly+quarterly)"
SHORT = {"A. Annual-only, revised": "A. Annual-only, revised lags",
         "B. Annual-only, real-time": "B. Annual-only, real-time lags",
         "C0. Mixed (monthly), real-time": "C0. + monthly GEM",
         FINAL: "C. PRISM final (+ quarterly GDP)",
         "GBM (HistGB)": "GBM preview (untuned, with Pink Sheet)",
         "AR(2) pooled": "AR(2) pooled", "Naive (real-time)": "Naive (real-time)", "IMF WEO": "IMF WEO"}


def save(df, name):
    df.to_csv(C.TAB_DIR / f"final_{name}.csv")
    return df


def combined(m, h):
    arms, _ = FN.rolling_arms(m, h, importance=(m, h) == (4, 0))
    lad = pd.read_pickle(C.OUT_DIR / "cache" / f"pred_m{m:02d}_h{h}.pkl")
    lad = lad[lad["model"].isin(["IMF WEO", "GBM (HistGB)", "AR(2) pooled", "Naive (real-time)"])]
    return pd.concat([arms, lad], ignore_index=True)


def main():
    t0 = time.time()
    # ------------------------------------------------------------ tuning
    sch = pd.concat([FN.tuned_schedule(4, h).assign(horizon=h) for h in (0, 1)])
    save(sch.round(3), "tuning_schedule")
    print(sch)

    # ------------------------------------------------------------ RQ1 + final ladder
    preds, rows, dms = {}, [], []
    for m, h in [(4, 0), (4, 1), (6, 0), (6, 1)]:
        p = combined(m, h)
        preds[(m, h)] = p
        for label, kw in [("2000-2022", dict(years=(2000, 2022))),
                          ("2000-2022 excl. 2020-21", dict(years=(2000, 2022), exclude_crisis=True))]:
            t = E.score_table(p, **kw).reset_index()
            t["cutoff"], t["h"], t["sample"] = m, h, label
            rows.append(t)
        for a, b in [(FINAL, "IMF WEO"), (FINAL, "AR(2) pooled"), (FINAL, "Naive (real-time)"),
                     ("B. Annual-only, real-time", "A. Annual-only, revised"),
                     (FINAL, "B. Annual-only, real-time"), (FINAL, "GBM (HistGB)")]:
            for loss in ("se", "ae"):
                bench_years = (2000, 2022) if "WEO" in b else (2000, 2024)
                r = FN.dm_any(p, a, b, h=h, loss=loss, years=bench_years)
                r.update(cutoff=m, h=h)
                dms.append(r)
    rq1 = pd.concat(rows, ignore_index=True)
    rq1["model"] = rq1["model"].map(SHORT).fillna(rq1["model"])
    save(rq1.round(3), "rq1_scores")
    dm = pd.DataFrame(dms)
    dm["model"] = dm["model"].map(SHORT)
    dm["bench"] = dm["bench"].map(SHORT)
    save(dm.round(4), "dm_tests")
    print(rq1[(rq1.cutoff == 4) & (rq1.h == 0)][["model", "sample", "n", "RMSE", "MAE", "RMSE_vs_naive"]])
    print(dm.round(3))

    # no-benchmark window 2000-2024 and rest of world
    extra = []
    for (m, h), p in preds.items():
        no = p[~p["model"].str.contains("WEO")]
        for lab, kw in [("South Asia 2000-2024", dict(years=(2000, 2024))),
                        ("Rest of world 2000-2024", dict(years=(2000, 2024), sa_only=False))]:
            d = no if "South" in lab else no[no["is_sa"] == 0]
            t = E.score_table(d, **kw).reset_index()
            t["cutoff"], t["h"], t["sample"] = m, h, lab
            extra.append(t)
    extra = pd.concat(extra)
    extra["model"] = extra["model"].map(SHORT)
    save(extra.round(3), "scores_2000_2024")

    # per country and effective horizon, April h=0
    p = preds[(4, 0)]
    pc = []
    for iso in C.SA:
        t = E.score_table(p[p["iso"] == iso], models=[FINAL, "IMF WEO", "AR(2) pooled", "Naive (real-time)"],
                          years=(2000, 2022))
        pc.append(t["RMSE"].rename(C.SA_NAMES[iso]))
    pc = pd.DataFrame(pc).rename(columns=SHORT)
    save(pc.round(2), "rmse_by_country_m04_h0")
    print(pc.round(2))

    # ------------------------------------------------------------ importance
    _, imp = FN.rolling_arms(4, 0, importance=True)
    it = FN.importance_table(imp)
    save(it.round(4), "permutation_importance_m04_h0")
    fig_importance(it)

    # ------------------------------------------------------------ RQ2
    ivs = []
    for m, h in [(4, 0), (4, 1), (6, 0)]:
        p = preds[(m, h)]
        store = FN.load_final_store(m, h)
        for meth in ("global", "scaled", "adaptive"):
            ivs.append(E.conformal(p, FINAL, method=meth).assign(horizon=h, cutoff=m))
        ivs.append(FN.crisis_conformal(p, store, FINAL).assign(horizon=h, cutoff=m))
    rq2 = []
    for m, h in [(4, 0), (4, 1), (6, 0)]:
        t = FN.rq2_table([iv for iv in ivs if iv["horizon"].iloc[0] == h and iv["cutoff"].iloc[0] == m])
        t["h"], t["cutoff"] = h, m
        rq2.append(t)
    rq2 = pd.concat(rq2, ignore_index=True)
    save(rq2.round(3), "rq2_intervals")
    pd.concat(ivs).to_pickle(C.OUT_DIR / "cache" / "final_intervals.pkl")
    print(rq2[rq2.h == 0].round(1).to_string())
    fig_rq2(rq2[(rq2.h == 0) & (rq2.cutoff == 4)], rq2[(rq2.h == 0) & (rq2.cutoff == 6)])

    # ------------------------------------------------------------ monthly refresh of the final model
    ref = []
    for m in range(1, 13):
        a, _ = FN.rolling_arms(m, 0, arms=("C",))
        ref.append(a)
    ref = pd.concat(ref)
    base = pd.read_pickle(C.OUT_DIR / "cache" / "refresh_h0.pkl")
    allp = pd.concat([ref, base[base["model"].isin(["IMF WEO", "AR(2) pooled", "Naive (real-time)"])]])
    rows = []
    for m, g in allp.groupby("cutoff_month"):
        r = E.score_table(g, years=(2000, 2022), ref=None)["RMSE"].rename(m)
        rows.append(r)
    rf = pd.DataFrame(rows).rename(columns={FINAL: "PRISM final"})
    rf.index.name = "cutoff_month"
    save(rf.round(3), "monthly_refresh")
    print(rf.round(2))
    fig_refresh(rf)

    # ------------------------------------------------------------ ladder figure
    t = rq1[(rq1.cutoff == 4) & (rq1.h == 0)]
    fig_ladder(t)
    print(f"done in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ figures
def fig_ladder(t):
    order = ["IMF WEO", "C. PRISM final (+ quarterly GDP)", "C0. + monthly GEM", "B. Annual-only, real-time lags",
             "GBM preview (untuned, with Pink Sheet)", "A. Annual-only, revised lags", "AR(2) pooled", "Naive (real-time)"]
    a = t[t["sample"] == "2000-2022"].set_index("model")["RMSE"].reindex(order)
    b = t[t["sample"] != "2000-2022"].set_index("model")["RMSE"].reindex(order)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    y = np.arange(len(order))[::-1]
    ax.barh(y + 0.19, a, 0.36, color=NAVY, label="All years")
    ax.barh(y - 0.19, b, 0.36, color="#e8a07d", label="Excluding 2020-21")
    for yy, v in zip(y, a):
        ax.text(v + 0.08, yy + 0.19, f"{v:.2f}", va="center", fontsize=8.5)
    for yy, v in zip(y, b):
        ax.text(v + 0.08, yy - 0.19, f"{v:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(y, order)
    ax.set_xlabel("RMSE, percentage points")
    ax.set_title("South Asia, April cutoff, current year (h = 0), 2000-2022")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    ax.set_xlim(0, 9.4)
    ax.grid(axis="y", visible=False)
    fig.savefig(C.FIG_DIR / "final_fig1_ladder.png")
    plt.close(fig)


def fig_importance(it):
    g = it[it.index.str.startswith("__group__")]["d_se_all"].rename(lambda s: s[9:])
    labels = {"growth": "Growth history", "annual": "Annual WDI structure", "monthly": "Monthly GEM",
              "quarterly": "Quarterly GDP"}
    nice = {"g_lag1_rt": "Growth last year (real-time)", "log_gdp_pc_a": "Log GDP per capita",
            "g_mean3_rt": "3-year mean growth", "energy_use_a": "Energy use per capita",
            "ip_fytd": "Industrial prod., fiscal YTD", "q_yoy_last": "Quarterly GDP, latest y/y",
            "exports_m_fytd": "Exports, fiscal YTD", "manuf_share_a": "Manufacturing share",
            "impcov_m_lvl3": "Import cover (months)", "fdi_a": "FDI, % GDP", "investment_a": "Investment, % GDP",
            "reserves_m_yoy3": "Reserves, 3m y/y", "g_lag2_rt": "Growth two years ago",
            "world_g_lag1": "World growth last year", "exports_m_yoy3": "Exports, 3m y/y",
            "ip_yoy3": "Industrial prod., 3m y/y", "g_lag3": "Growth three years ago"}
    f = it[~it.index.str.startswith("__group__")].head(12)["d_se_all"][::-1]
    f.index = [nice.get(i, i) for i in f.index]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.5]})
    gg = g.sort_values()
    axs[0].barh([labels[k] for k in gg.index], gg, color=BLUE)
    axs[0].set_title("By feature group")
    axs[0].set_xlabel("Increase in MSE when shuffled (pp²)")
    axs[1].barh(f.index, f, color=AQUA)
    axs[1].set_title("Top 12 features")
    axs[1].set_xlabel("Increase in MSE when shuffled (pp²)")
    for a in axs:
        a.grid(axis="y", visible=False)
    fig.suptitle("Permutation importance, PRISM final, April cutoff, h = 0, all economies 2000-2024",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "final_fig2_importance.png")
    plt.close(fig)


def fig_rq2(t4, t6):
    meths = ["global", "scaled", "adaptive", "crisis-aware"]
    samples = ["South Asia, excl. 2020-21", "South Asia, 2020-21", "Global, 2009", "Global, 2020"]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    w = 0.2
    cols = [GREY, BLUE, AQUA, ORANGE]
    for ax, t, title in [(axs[0], t4, "April cutoff (data to Feb/Mar)"), (axs[1], t6, "June cutoff (data to Apr/May)")]:
        for i, mth in enumerate(meths):
            sub = t[t.method == mth].set_index("sample").reindex(samples)
            x = np.arange(len(samples)) + (i - 1.5) * w
            ax.bar(x, sub["coverage_%"], w, color=cols[i], label=mth)
        ax.axhline(80, color=INK, lw=1, ls="--")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(samples)), [s.replace("South Asia, ", "SA ") for s in samples], fontsize=8.5)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("Coverage of 80% interval, %")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Interval coverage, PRISM final, current year (dashed = nominal 80%)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "final_fig3_intervals.png")
    plt.close(fig)


def fig_refresh(rf):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.plot(rf.index, rf["PRISM final"], marker="o", color=BLUE, lw=2.2, label="PRISM final")
    for c, col, lab in [("AR(2) pooled", GREY, "AR(2) pooled"), ("IMF WEO", ORANGE, "IMF WEO (latest round)")]:
        if c in rf:
            ax.plot(rf.index, rf[c], marker="o", color=col, lw=1.8, label=lab, ls="--" if "WEO" in c else "-")
    ax.set_xticks(range(1, 13), months)
    ax.set_ylabel("RMSE, pp")
    ax.set_title("Current-year RMSE by month-end cutoff, South Asia, 2000-2022")
    ax.legend(frameon=False)
    fig.savefig(C.FIG_DIR / "final_fig4_refresh.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
