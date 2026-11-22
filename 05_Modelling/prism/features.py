"""Vintage-aware feature store.

build_store(cutoff_month, horizon) returns one row per (iso, origin year t):
the features a forecaster could have held at the end of `cutoff_month` in
year t, and the realised growth of the target year T = t + horizon.

Rules (this replaces the imputation of 02_Prepare_Data.ipynb):
  * Annual WDI values for year t-1 are public only from July of year t.
    Before that the latest public year is t-2.
  * A value is carried forward within its own country for at most
    MAX_ANNUAL_STALENESS years, and only from years that were already public.
  * No cross-sectional filling. The income-group median used in the EDA is
    not used here: it mixes in other countries' values for the same year,
    including years a forecaster had not yet seen.
  * Any remaining gap stays NaN. Models that cannot take NaN fill it inside
    the training fold (median of the training rows) and get a 0/1 flag, so
    the fill never uses test-period data.
  * Monthly GEM and Pink Sheet values are used only up to the last month
    published by the cutoff (config.MONTHLY_LAG).
"""
import numpy as np
import pandas as pd

from . import config as C
from . import data as D

GEM_LEVEL = {          # indicator -> short name; growth computed as y/y of 3-month average
    "IPTOTSAKD": "ip",
    "DXGSRMRCHSACD": "exports_m",
    "DMGSRMRCHSACD": "imports_m",
    "TOTRESV": "reserves_m",
    "CPTOTSAXN": "cpi_m",
    "DPANUSSPB": "fx_m",
    "REER": "reer_m",
}
GEM_RATIO = {"IMPCOV": "impcov_m"}   # used as a level (3-month average)
PINK = ["pink_energy", "pink_nonenergy", "pink_food", "pink_metals"]


# --------------------------------------------------------------- helpers
def _latest_weo_vintage(t, m):
    """Most recent WEO round public at the end of month m of year t."""
    if m >= 10:
        return t, "F"
    if m >= 4:
        return t, "S"
    return t - 1, "F"


def _realtime_growth(index, m, back=1):
    """Real-time estimate of growth in year t-back, as printed in the latest
    WEO round available at the cutoff (each WEO round prints the two most
    recent historical years). NaN where WEO does not cover the year."""
    w = D.load_weo()
    w = w.set_index(["iso", "vintage_year", "season", "target_year"])["value"]
    w = w[~w.index.duplicated()]
    out = []
    for iso, t in index:
        vy, s = _latest_weo_vintage(t, m)
        out.append(w.get((iso, vy, s, t - back), np.nan))
    return np.array(out, dtype=float)


def _annual_block(wdi, cols, origins, m):
    """Latest public value of each annual WDI feature at the cutoff,
    carried forward within country for at most MAX_ANNUAL_STALENESS years."""
    avail_year = origins["year"] - np.where(m >= C.ANNUAL_RELEASE_MONTH, 1, 2)
    panel = wdi.set_index(["iso", "year"])[cols].sort_index()
    # forward-fill within country with a limit: a value from year y can stand
    # in for year y+1 and y+2 only. Because we look it up at avail_year, it is
    # never a value from after the cutoff.
    filled = panel.groupby(level="iso").ffill(limit=C.MAX_ANNUAL_STALENESS)
    key = pd.MultiIndex.from_arrays([origins["iso"], avail_year])
    block = filled.reindex(key).reset_index(drop=True)
    block.columns = [c + "_a" for c in cols]
    return block


def _monthly_panel():
    """Monthly GEM indicators on a complete (iso, month) grid, plus derived
    3-month-average series."""
    g = D.load_gem()
    isos = g.index.get_level_values("iso").unique()
    periods = pd.period_range(g.index.get_level_values("period").min(),
                              g.index.get_level_values("period").max(), freq="M")
    g = g.reindex(pd.MultiIndex.from_product([isos, periods], names=["iso", "period"]))
    return g


def _monthly_block(origins, m, horizon):
    g = _monthly_panel()
    out = {}
    cut = pd.Period(year=2000, month=m, freq="M")
    for code, name in {**GEM_LEVEL, **GEM_RATIO}.items():
        if code not in g.columns:
            continue
        s = g[code]
        r3 = s.groupby(level="iso").transform(lambda x: x.rolling(3, min_periods=3).mean())
        if code in GEM_LEVEL:
            feat = (r3 / r3.groupby(level="iso").shift(12) - 1) * 100
            fname = name + "_yoy3"
        else:
            feat = r3
            fname = name + "_lvl3"
        lag = C.MONTHLY_LAG.get(code, 2)
        last = [pd.Period(year=int(t), month=m, freq="M") - lag for t in origins["year"]]
        key = pd.MultiIndex.from_arrays([origins["iso"], pd.PeriodIndex(last, freq="M")])
        out[fname] = feat.reindex(key).to_numpy()

        # fiscal-year-to-date growth of the TARGET reporting year (h = 0 only):
        # average of the months of the target year published by the cutoff,
        # against the same months a year earlier. Aligns monthly data with
        # each country's own reporting period.
        if code in ("IPTOTSAKD", "DXGSRMRCHSACD", "DMGSRMRCHSACD") and horizon == 0:
            vals = np.full(len(origins), np.nan)
            for i, (iso, t) in enumerate(zip(origins["iso"], origins["year"])):
                (sy, sm), _ = C.fy_window(iso, int(t))
                start = pd.Period(year=sy, month=sm, freq="M")
                stop = last[i]
                if stop < start:
                    continue
                try:
                    cur = s.loc[iso].loc[start:stop]
                    prev = s.loc[iso].loc[start - 12:stop - 12]
                except KeyError:
                    continue
                if cur.notna().all() and prev.notna().all() and len(cur) == len(prev) > 0:
                    vals[i] = (cur.mean() / prev.mean() - 1) * 100
            out[name + "_fytd"] = vals
    return pd.DataFrame(out)


def _pink_block(origins, m):
    p = D.load_pinksheet()[PINK]
    r3 = p.rolling(3, min_periods=3).mean()
    yoy = (r3 / r3.shift(12) - 1) * 100
    last = pd.PeriodIndex([pd.Period(year=int(t), month=m, freq="M") - C.PINK_LAG
                           for t in origins["year"]], freq="M")
    block = yoy.reindex(last).reset_index(drop=True)
    block.columns = [c + "_yoy3" for c in PINK]
    return block


# --------------------------------------------------------------- main entry
def build_store(cutoff_month=4, horizon=0, first_origin=1962, last_origin=2024):
    """Feature store for one information cutoff and one horizon."""
    wdi = D.load_wdi()
    m = int(cutoff_month)
    g = wdi.set_index(["iso", "year"])["gdp_growth"]

    isos = wdi["iso"].unique()
    origins = pd.DataFrame([(i, t) for i in isos for t in range(first_origin, last_origin + 1)],
                           columns=["iso", "year"])
    origins["target_year"] = origins["year"] + horizon
    origins["y"] = g.reindex(pd.MultiIndex.from_arrays([origins["iso"], origins["target_year"]])).to_numpy()

    # ---- growth history known at the cutoff
    last_g = origins["year"] - np.where(m >= C.GDP_PRELIM_MONTH, 1, 2)
    for k in (1, 2, 3):
        key = pd.MultiIndex.from_arrays([origins["iso"], last_g - (k - 1)])
        origins[f"g_lag{k}"] = g.reindex(key).to_numpy()
    # Real-time (as-published) values of the last two years. These are what a
    # forecaster actually saw; the WDI numbers above are later revisions.
    # With m < 3 the year t-1 is not yet known, so the real-time lags shift too.
    back = 1 if m >= C.GDP_PRELIM_MONTH else 2
    idx = list(zip(origins["iso"], origins["year"]))
    origins["g_lag1_rt"] = _realtime_growth(idx, m, back)
    origins["g_lag2_rt"] = _realtime_growth(idx, m, back + 1)
    origins["rt_available"] = origins["g_lag1_rt"].notna().astype(int)
    # where WEO has no real-time number (before 1990, after 2022, or an
    # economy WEO does not cover), fall back to the WDI value
    origins["g_lag1_rt"] = origins["g_lag1_rt"].fillna(origins["g_lag1"])
    origins["g_lag2_rt"] = origins["g_lag2_rt"].fillna(origins["g_lag2"])
    origins["g_mean3_rt"] = origins[["g_lag1_rt", "g_lag2_rt", "g_lag3"]].mean(axis=1)
    origins["g_mean3"] = origins[["g_lag1", "g_lag2", "g_lag3"]].mean(axis=1)

    # global cycle: median growth of all countries in the last known year
    world = g.groupby(level="year").median()
    origins["world_g_lag1"] = world.reindex(last_g).to_numpy()

    # ---- annual structure (WDI), publication-lagged
    cols = [c for c in D.wdi_feature_cols(wdi) if c != "exchange_rate"]
    annual = _annual_block(wdi, cols, origins, m)
    # exchange-rate change instead of its level (levels are not comparable across countries)
    fx = wdi.set_index(["iso", "year"])["exchange_rate"]
    fx_chg = fx.groupby(level="iso").pct_change(fill_method=None) * 100
    avail_year = origins["year"] - np.where(m >= C.ANNUAL_RELEASE_MONTH, 1, 2)
    annual["fx_change_a"] = fx_chg.reindex(pd.MultiIndex.from_arrays([origins["iso"], avail_year])).to_numpy()

    monthly = _monthly_block(origins, m, horizon)
    pink = _pink_block(origins, m)

    meta = wdi.drop_duplicates("iso").set_index("iso")[["income_groups", "world_6region", "name"]]
    store = pd.concat([origins.reset_index(drop=True), annual, monthly, pink], axis=1)
    store = store.join(meta, on="iso")
    store["is_sa"] = store["iso"].isin(C.SA).astype(int)
    store["cutoff_month"] = m
    store["horizon"] = horizon
    store["months_ahead"] = [C.months_to_fy_end(i, ty, m, t)
                             for i, ty, t in zip(store["iso"], store["target_year"], store["year"])]
    return store


def feature_groups(store):
    # real-time growth history (default) ...
    growth = ["g_lag1_rt", "g_lag2_rt", "g_lag3", "g_mean3_rt", "world_g_lag1"]
    # ... and the revised-data version, used only to measure the vintage effect
    growth_revised = ["g_lag1", "g_lag2", "g_lag3", "g_mean3", "world_g_lag1"]
    annual = [c for c in store.columns if c.endswith("_a")]
    monthly = [c for c in store.columns if c.endswith(("_yoy3", "_lvl3", "_fytd")) and not c.startswith("pink")]
    pink = [c for c in store.columns if c.startswith("pink")]
    return {"growth": growth, "annual": annual, "monthly": monthly, "pink": pink,
            "growth_revised": growth_revised}
