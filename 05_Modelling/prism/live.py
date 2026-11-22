"""Live (2026) feature store and the deployment bundle.

The research store stops at origin 2024 because every row needs a realised
outcome. The deployed model forecasts origin 2026, which needs three patches,
each using only information public at the cutoff:

  * WDI ends in 2024, so last year's growth (2025) is taken from the World
    Bank GEP June 2026 estimate, the real-time figure a forecaster held in
    mid-2026. Only cutoffs from June onwards are offered for that reason.
  * The annual panel is extended with empty 2025-2026 rows so the two-year
    forward fill reaches the live origin.
  * World growth in the last known year (2025) is the median GEP estimate.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import features as F
from . import final as FN

LIVE_ORIGIN = 2026
LIVE_CUTOFFS = (6, 8)     # June 2026 (GEP round) and August 2026 (latest month every
                          # monthly series is published under the lag rules)


_ORIG_LOAD_WDI = D.load_wdi


@lru_cache(maxsize=None)
def _wdi_extended():
    w = _ORIG_LOAD_WDI()
    last = w["year"].max()
    meta = w.drop_duplicates("iso")[["iso", "name", "income_groups", "world_6region", "income_encoded"]]
    add = [meta.assign(year=y) for y in range(last + 1, LIVE_ORIGIN + 1)]
    return pd.concat([w] + add, ignore_index=True).sort_values(["iso", "year"]).reset_index(drop=True)


def gep_estimates():
    g = D.load_gep()
    return g.set_index(["iso", "year"])["gep_jun2026_growth"]


def build_live_store(m, h, cache=True):
    path = C.OUT_DIR / "cache" / f"live_m{m:02d}_h{h}.pkl"
    if cache and path.exists():
        return pd.read_pickle(path)
    orig = D.load_wdi
    F.D.load_wdi = _wdi_extended          # build_store reads the panel through data.load_wdi
    try:
        s = F.build_store(m, h, first_origin=1962, last_origin=LIVE_ORIGIN)
    finally:
        F.D.load_wdi = orig
    q = FN.quarterly_block(s[["iso", "year"]], m, h)
    s = pd.concat([s.reset_index(drop=True), q], axis=1)
    gep = gep_estimates()
    live = s["year"] == LIVE_ORIGIN
    for col, back in (("g_lag1_rt", 1), ("g_lag2_rt", 2)):
        key = list(zip(s.loc[live, "iso"], s.loc[live, "year"] - back))
        est = pd.Series([gep.get(k, np.nan) for k in key], index=s.index[live])
        s.loc[live, col] = s.loc[live, col].fillna(est)
    s.loc[live, "g_mean3_rt"] = s.loc[live, ["g_lag1_rt", "g_lag2_rt", "g_lag3"]].mean(axis=1)
    world25 = gep.xs(2025, level="year").median()
    s.loc[live & s["world_g_lag1"].isna(), "world_g_lag1"] = world25
    s["gep_2026_forecast"] = [gep.get((i, ty), np.nan) for i, ty in zip(s["iso"], s["target_year"])]
    path.parent.mkdir(exist_ok=True)
    s.to_pickle(path)
    return s
