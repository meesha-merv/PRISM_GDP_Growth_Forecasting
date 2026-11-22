"""The experiments behind the progress report. Each function saves its
tables to outputs/tables and returns them, so the notebook and the
command-line runner (run_experiments.py) produce identical results."""
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from . import evaluate as E
from . import features as F
from . import models as M
from . import reconcile as R

warnings.filterwarnings("ignore")

# Chart colours (validated categorical slots 1-3 + neutral reference ink)
BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#2b2b29"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#b9b8b2", "axes.labelcolor": INK,
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e7e6e1", "grid.linewidth": 0.8,
                     "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "savefig.dpi": 160, "savefig.bbox": "tight"})

WEO_WINDOW = (2000, C.WEO_LAST_VINTAGE)     # years where the IMF benchmark exists


def _save(df, name):
    df.to_csv(C.TAB_DIR / f"{name}.csv")
    return df


def model_ladder(store):
    fg = F.feature_groups(store)
    allf = fg["growth"] + fg["annual"] + fg["monthly"] + fg["pink"]
    annual_only = fg["growth"] + fg["annual"]
    return [
        M.Naive(),
        M.Naive("g_lag1", "Naive (revised data)"),
        M.CountryMean(),
        M.PooledAR(),
        M.FixedEffectsAR(),
        M.RidgeModel(annual_only, True, "Ridge + FE (annual only)"),
        M.RidgeModel(allf, True),
        M.RidgeModel(allf, False),
        M.GBM(allf),
        M.WEOBenchmark(),
        M.WEOCorrection(allf),
    ]


# ------------------------------------------------------------------ E1/E2: baseline ladder
def run_ladder(cutoff=4, horizon=0, verbose=False):
    path = C.OUT_DIR / "cache" / f"pred_m{cutoff:02d}_h{horizon}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    store = E.load_store(cutoff, horizon)
    test_years = [t for t in C.TEST_YEARS if t + horizon <= 2024]
    pred = E.rolling_origin(store, model_ladder(store), test_years, verbose=verbose)
    pred.to_pickle(path)
    return pred


def ladder_tables(pred, tag):
    out = {
        "sa_weo_window": E.score_table(pred, years=WEO_WINDOW),
        "sa_weo_window_excl_crisis": E.score_table(pred, years=WEO_WINDOW, exclude_crisis=True),
        "global_weo_window": E.score_table(pred, sa_only=False, years=WEO_WINDOW),
    }
    no_weo = [m for m in pred["model"].unique() if "WEO" not in m]
    out["sa_2000_2024_no_benchmark"] = E.score_table(pred, models=no_weo, years=(2000, 2024))
    for k, v in out.items():
        _save(v.round(3), f"{tag}_{k}")
    return out


def per_country(pred, models=("Naive (real-time)", "AR(2) pooled", "GBM (HistGB)", "LightGBM", "IMF WEO"), tag="m04_h0"):
    d = pred[(pred["is_sa"] == 1) & pred["target_year"].between(*WEO_WINDOW) & pred["model"].isin(models)]
    w = d.pivot_table(index=["iso", "target_year"], columns="model", values="pred").dropna()
    y = d.drop_duplicates(["iso", "target_year"]).set_index(["iso", "target_year"])["y"].reindex(w.index)
    rmse = w.sub(y, axis=0).pow(2).groupby(level="iso").mean().pow(0.5)
    rmse["n"] = w.groupby(level="iso").size()
    rmse.index = [C.SA_NAMES[i] for i in rmse.index]
    return _save(rmse.round(2), f"{tag}_rmse_by_country")


def by_effective_horizon(pred, models=("Naive (real-time)", "GBM (HistGB)", "LightGBM", "IMF WEO"), tag="m04_h0"):
    d = pred[(pred["is_sa"] == 1) & pred["target_year"].between(*WEO_WINDOW) & pred["model"].isin(models)].copy()
    d["horizon_group"] = pd.cut(d["months_ahead"], [-1, 4, 9, 13, 18, 24],
                                labels=["0-4 mo", "5-9 mo", "10-13 mo", "14-18 mo", "19-24 mo"])
    w = d.pivot_table(index=["iso", "target_year", "horizon_group"], columns="model", values="pred", observed=True).dropna()
    y = d.drop_duplicates(["iso", "target_year"]).set_index(["iso", "target_year"])["y"]
    y = y.reindex(w.index.droplevel("horizon_group")).to_numpy()
    err = w.sub(y, axis=0).pow(2)
    t = err.groupby(level="horizon_group", observed=True).mean().pow(0.5)
    t["n"] = err.groupby(level="horizon_group", observed=True).size()
    t["countries"] = d.groupby("horizon_group", observed=True)["iso"].agg(lambda s: ", ".join(sorted(s.unique())))
    return _save(t.round(2), f"{tag}_rmse_by_effective_horizon")


def dm_table(pred, horizon, tag):
    rows = [E.dm_test(pred[pred["target_year"].between(*WEO_WINDOW)], m, h=horizon)
            for m in pred["model"].unique() if m != "IMF WEO"]
    rows += [E.dm_test(pred[pred["target_year"].between(*WEO_WINDOW)], m, bench="Naive (real-time)", h=horizon)
             for m in pred["model"].unique() if m != "Naive (real-time)"]
    return _save(pd.DataFrame(rows).round(4), f"{tag}_dm_tests")


# ------------------------------------------------------------------ E4: ablation
def ablation(cutoff=4, horizon=0):
    path = C.OUT_DIR / "cache" / f"ablation_m{cutoff:02d}_h{horizon}.pkl"
    if path.exists():
        pred = pd.read_pickle(path)
    else:
        store = E.load_store(cutoff, horizon)
        fg = F.feature_groups(store)
        sets = {
            "growth history only": fg["growth"],
            "+ annual WDI": fg["growth"] + fg["annual"],
            "+ annual + monthly GEM": fg["growth"] + fg["annual"] + fg["monthly"],
            "+ annual + monthly + Pink Sheet": fg["growth"] + fg["annual"] + fg["monthly"] + fg["pink"],
            "all features, REVISED growth lags": fg["growth_revised"] + fg["annual"] + fg["monthly"] + fg["pink"],
        }
        mods = []
        for label, cols in sets.items():
            mods.append(M.RidgeModel(cols, False, f"Ridge | {label}"))
            mods.append(M.GBM(cols, name=f"GBM | {label}"))
        test_years = [t for t in C.TEST_YEARS if t + horizon <= 2024]
        pred = E.rolling_origin(store, mods, test_years)
        pred.to_pickle(path)
    rows = []
    for sample, kw in [("South Asia", dict(sa_only=True)), ("Rest of world", dict(sa_only=False))]:
        d = pred if sample == "South Asia" else pred[pred["is_sa"] == 0]
        t = E.score_table(d, years=(2000, 2024), ref=None, **kw)[["n", "RMSE", "MAE"]]
        t["sample"] = sample
        rows.append(t)
    out = pd.concat(rows).reset_index()
    out["family"] = out["model"].str.split(" | ").str[0]
    out["features"] = out["model"].str.split(r" \| ").str[1]
    out = out.pivot_table(index="features", columns=["sample", "family"], values="RMSE")
    order = ["growth history only", "+ annual WDI", "+ annual + monthly GEM",
             "+ annual + monthly + Pink Sheet", "all features, REVISED growth lags"]
    return _save(out.reindex(order).round(3), f"m{cutoff:02d}_h{horizon}_ablation")


# ------------------------------------------------------------------ E5: intervals
def interval_experiment(pred, models, tag):
    ivs = [E.conformal(pred, m, method=meth) for m in models for meth in ("global", "scaled", "adaptive")]
    iv = pd.concat(ivs, ignore_index=True)
    iv.to_pickle(C.OUT_DIR / "cache" / f"intervals_{tag}.pkl")
    cov = E.coverage_table(iv[iv["target_year"] >= 2000])
    return iv, _save(cov.round(2), f"{tag}_interval_coverage")


# ------------------------------------------------------------------ E6: monthly refresh
def monthly_refresh(months=range(1, 13), horizon=0):
    path = C.OUT_DIR / "cache" / f"refresh_h{horizon}.pkl"
    if path.exists():
        pred = pd.read_pickle(path)
    else:
        parts = []
        for m in months:
            store = E.load_store(m, horizon)
            fg = F.feature_groups(store)
            allf = fg["growth"] + fg["annual"] + fg["monthly"] + fg["pink"]
            mods = [M.Naive(), M.PooledAR(), M.RidgeModel(allf, False), M.GBM(allf), M.WEOBenchmark()]
            parts.append(E.rolling_origin(store, mods, [t for t in C.TEST_YEARS if t <= 2024]))
        pred = pd.concat(parts, ignore_index=True)
        pred.to_pickle(path)
    rows = []
    for m, g in pred.groupby("cutoff_month"):
        no_weo = g[g["model"] != "IMF WEO"]
        s = E.score_table(no_weo, years=(2000, 2024), ref=None)["RMSE"].rename(m)
        w = E.score_table(g, years=WEO_WINDOW, ref=None)["RMSE"].add_suffix(" [2000-22]").rename(m)
        rows.append(pd.concat([s, w]))
    return pred, _save(pd.DataFrame(rows).round(3), f"refresh_h{horizon}_rmse_by_cutoff")


# ------------------------------------------------------------------ E8: which truth?
def truth_robustness(pred, tag="m04_h0"):
    """Score against the first published estimate (WEO, a year later)
    instead of today's revised WDI value."""
    hist = R.weo_history(lag=1, season="S")
    d = pred.copy()
    d["y"] = hist.reindex(pd.MultiIndex.from_arrays([d["iso"], d["target_year"]])).to_numpy()
    d = d.dropna(subset=["y"])
    return _save(E.score_table(d, years=WEO_WINDOW).round(3), f"{tag}_scores_vs_first_release")


# ------------------------------------------------------------------ figures
def fig_ladder(tab_all, tab_ex, fname="fig1_baseline_ladder.png", title="South Asia, April cutoff, current year (h = 0), 2000-2022"):
    order = tab_all.sort_values("RMSE", ascending=False).index
    fig, ax = plt.subplots(figsize=(8, 5))
    yy = np.arange(len(order))
    ax.barh(yy + 0.2, tab_all.loc[order, "RMSE"], height=0.38, color=BLUE, label="All years")
    ax.barh(yy - 0.2, tab_ex.loc[order, "RMSE"], height=0.38, color=ORANGE, label="Excluding 2020-21")
    for i, m in enumerate(order):
        ax.text(tab_all.loc[m, "RMSE"] + 0.08, i + 0.2, f"{tab_all.loc[m, 'RMSE']:.2f}", va="center", fontsize=8, color=INK)
        ax.text(tab_ex.loc[m, "RMSE"] + 0.08, i - 0.2, f"{tab_ex.loc[m, 'RMSE']:.2f}", va="center", fontsize=8, color="#52514e")
    ax.set_yticks(yy, order)
    ax.set_xlabel("RMSE, percentage points of GDP growth (lower is better)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=2, frameon=False)
    fig.savefig(C.FIG_DIR / fname)
    plt.close(fig)
    return C.FIG_DIR / fname


def fig_refresh(tab, gbm_name, fname="fig2_monthly_refresh.png"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = tab.index.to_numpy()
    for col, c, lab in [("Naive (real-time) [2000-22]", GREY, "Naive (real-time)"),
                        ("Ridge pooled [2000-22]", AQUA, "Ridge pooled"),
                        (f"{gbm_name} [2000-22]", BLUE, gbm_name),
                        ("IMF WEO [2000-22]", ORANGE, "IMF WEO (latest round)")]:
        if col in tab:
            ax.plot(x, tab[col], color=c, lw=2, marker="o", ms=5, label=lab)
            ax.annotate(lab, (x[-1], tab[col].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
    ax.set_xticks(x, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(x)])
    ax.axvline(4, color="#b9b8b2", lw=1, ls="--"); ax.axvline(10, color="#b9b8b2", lw=1, ls="--")
    ax.text(4.1, ax.get_ylim()[1] * 0.97, "WEO spring", fontsize=8, color="#52514e", va="top")
    ax.text(10.1, ax.get_ylim()[1] * 0.97, "WEO fall", fontsize=8, color="#52514e", va="top")
    ax.set_xlabel("Month-end information cutoff in the target year")
    ax.set_ylabel("RMSE, pp (South Asia, 2000-2022)")
    ax.set_title("Current-year forecast error as the year unfolds", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_xlim(0.6, 13.8)
    fig.savefig(C.FIG_DIR / fname)
    plt.close(fig)
    return C.FIG_DIR / fname


def fig_country_paths(pred, iv, gbm_name, fname="fig3_country_forecasts.png"):
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
    d = pred[(pred["is_sa"] == 1)]
    for ax, iso in zip(axs.flat, C.SA):
        a = d[(d["iso"] == iso) & (d["model"] == "Naive (real-time)")].sort_values("target_year")
        g = d[(d["iso"] == iso) & (d["model"] == gbm_name)].sort_values("target_year")
        w = d[(d["iso"] == iso) & (d["model"] == "IMF WEO")].dropna(subset=["pred"]).sort_values("target_year")
        b = iv[(iv["iso"] == iso)].sort_values("target_year")
        ax.fill_between(b["target_year"], b["lo"], b["hi"], color=BLUE, alpha=0.15, lw=0, label="80% interval (scaled conformal)")
        ax.plot(a["target_year"], a["y"], color=INK, lw=2, label="Actual (WDI)")
        ax.plot(g["target_year"], g["pred"], color=BLUE, lw=2, label=gbm_name)
        ax.plot(w["target_year"], w["pred"], color=ORANGE, lw=2, ls="--", label="IMF WEO (April)")
        ax.set_title(C.SA_NAMES[iso], loc="left")
        ax.axhline(0, color="#b9b8b2", lw=0.8)
    axs[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.supylabel("Real GDP growth, %")
    fig.suptitle("April-cutoff, current-year forecasts by country (rolling origin)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / fname)
    plt.close(fig)
    return C.FIG_DIR / fname


def fig_coverage(cov, fname="fig4_interval_coverage.png", model=None):
    c = cov[cov["model"] == model] if model else cov
    samples = ["South Asia, excl. 2020-21", "South Asia, 2020-21 only", "South Asia, all years", "Rest of world"]
    methods = ["global", "scaled", "adaptive"]
    colors = {"global": GREY, "scaled": BLUE, "adaptive": AQUA}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(samples))
    for i, meth in enumerate(methods):
        v = c[c["method"] == meth].set_index("sample").reindex(samples)
        ax.bar(x + (i - 1) * 0.26, v["coverage_%"], width=0.24, color=colors[meth], label=meth)
        for xi, val in zip(x + (i - 1) * 0.26, v["coverage_%"]):
            ax.text(xi, val + 1, f"{val:.0f}", ha="center", fontsize=8, color=INK)
    ax.axhline(80, color=ORANGE, lw=1.5, ls="--")
    ax.text(1.0, 82, "nominal 80%", color=INK, fontsize=8, ha="center")
    ax.set_xticks(x, samples, fontsize=8.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Empirical coverage, %")
    ax.set_title(f"80% interval coverage by conformal method ({model})", loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.savefig(C.FIG_DIR / fname)
    plt.close(fig)
    return C.FIG_DIR / fname


def fig_horizon(tab, gbm_name, fname="fig5_effective_horizon.png"):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(tab))
    for i, (col, c) in enumerate([("Naive (real-time)", GREY), (gbm_name, BLUE), ("IMF WEO", ORANGE)]):
        ax.bar(x + (i - 1) * 0.26, tab[col], width=0.24, color=c, label=col)
    ax.set_xticks(x, [f"{h}\n({c})" for h, c in zip(tab.index.astype(str), tab["countries"])], fontsize=8)
    ax.set_ylabel("RMSE, pp")
    ax.set_title("Error by months left in the target reporting year (April cutoff, h = 0)", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.savefig(C.FIG_DIR / fname)
    plt.close(fig)
    return C.FIG_DIR / fname
