"""Loaders for the five raw sources. Nothing here fills or imputes values:
every loader returns the data as published, with gaps left as NaN.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C

# Same readable names as 01_Acquire_Data.ipynb, so the EDA and the models agree.
WDI_NAMES = {
    "ny_gdp_mktp_kd_zg": "gdp_growth",        "ny_gdp_pcap_kd": "gdp_pc",
    "ne_gdi_totl_zs": "investment",           "ne_gdi_ftot_zs": "fixed_investment",
    "ne_con_prvt_zs": "private_consumption",  "ne_con_govt_zs": "govt_consumption",
    "nv_agr_totl_zs": "agri_share",           "nv_ind_totl_zs": "industry_share",
    "nv_ind_manf_zs": "manuf_share",          "nv_srv_totl_zs": "services_share",
    "ne_trd_gnfs_zs": "trade",                "ne_exp_gnfs_zs": "exports",
    "ne_imp_gnfs_zs": "imports",              "bn_cab_xoka_gd_zs": "current_account",
    "bx_trf_pwkr_dt_gd_zs": "remittances",    "bx_klt_dinv_wd_gd_zs": "fdi",
    "fi_res_totl_mo": "reserves_months",      "pa_nus_fcrf": "exchange_rate",
    "dt_tds_dect_ex_zs": "debt_service",      "fp_cpi_totl_zg": "inflation_cpi",
    "ny_gdp_defl_kd_zg": "inflation_defl",    "fr_inr_rinr": "real_interest",
    "fs_ast_prvt_gd_zs": "private_credit",    "gc_dod_totl_gd_zs": "govt_debt",
    "sp_pop_grow": "pop_growth",              "sp_urb_totl_in_zs": "urban_share",
    "sp_pop_dpnd": "dependency_ratio",        "eg_use_pcap_kg_oe": "energy_use",
    "sl_tlf_cact_zs": "labour_participation",
}
# Dropped exactly as in 02_Prepare_Data.ipynb: sizes, a near-copy of the
# target, and govt_debt (88% missing).
WDI_DROP = ["ny_gdp_mktp_kd", "sp_pop_totl", "ny_gdp_pcap_kd_zg", "gc_dod_totl_gd_zs"]


@lru_cache(maxsize=None)
def load_wdi():
    """Annual WDI panel, one row per (iso, year), NOT filled.
    Keeps only economies with a World Bank income group (as in Part 2)."""
    frames = []
    for f in sorted(C.WDI_DIR.glob("*.csv")):
        code = f.stem
        if code == "geo" or code in WDI_DROP:
            continue
        d = pd.read_csv(f)
        d.columns = ["iso", "year", code]
        d["iso"] = d["iso"].str.upper()
        frames.append(d.set_index(["iso", "year"]))
    wdi = pd.concat(frames, axis=1).reset_index().rename(columns=WDI_NAMES)

    geo = pd.read_csv(C.WDI_DIR / "geo.csv")
    geo = geo[geo["iso3166_1_alpha3"].notna()]
    meta = geo[["iso3166_1_alpha3", "name", "income_groups", "world_6region"]].rename(
        columns={"iso3166_1_alpha3": "iso"})
    meta["world_6region"] = meta["world_6region"].fillna("america")   # MAF, SXM (Part 2)
    wdi = wdi.merge(meta, on="iso", how="left")
    wdi = wdi[wdi["income_groups"].notna()].copy()
    wdi["log_gdp_pc"] = np.log(wdi["gdp_pc"])
    wdi["inflation_slog"] = np.sign(wdi["inflation_cpi"]) * np.log1p(wdi["inflation_cpi"].abs())
    income_order = {"low_income": 0, "lower_middle_income": 1,
                    "upper_middle_income": 2, "high_income": 3}
    wdi["income_encoded"] = wdi["income_groups"].map(income_order)
    return wdi.sort_values(["iso", "year"]).reset_index(drop=True)


def wdi_feature_cols(wdi=None, max_missing=0.90):
    """Numeric WDI features, after the >90%-missing rule. The share is
    measured on the rows that have a target, as in Part 2."""
    wdi = load_wdi() if wdi is None else wdi
    skip = {"year", "gdp_growth", "gdp_pc", "inflation_cpi", "income_encoded"}
    cols = [c for c in wdi.select_dtypes("number").columns if c not in skip]
    has_y = wdi[wdi["gdp_growth"].notna()]
    return [c for c in cols if has_y[c].isna().mean() <= max_missing]


@lru_cache(maxsize=None)
def load_gem():
    """GEM monthly indicators, wide: index (iso, period), one column per code."""
    g = pd.read_csv(C.RAW_DIR / "gem_monthly.csv")
    g["period"] = pd.PeriodIndex(g["date"].str.replace("M", "-"), freq="M")
    g = g.pivot_table(index=["iso", "period"], columns="indicator", values="value")
    return g.sort_index()


@lru_cache(maxsize=None)
def load_pinksheet():
    """Pink Sheet monthly price indices (2010=100): total, energy,
    non-energy, agriculture, food, fertilizers, metals."""
    x = pd.read_excel(C.RAW_DIR / "CMO-Historical-Data-Monthly.xlsx",
                      sheet_name="Monthly Indices", header=None)
    body = x.iloc[9:].copy()
    body = body[body[0].astype(str).str.match(r"^\d{4}M\d{2}$")]
    cols = {1: "pink_total", 2: "pink_energy", 3: "pink_nonenergy", 4: "pink_agri",
            6: "pink_food", 13: "pink_fertilizer", 14: "pink_metals"}
    out = body[[0] + list(cols)].rename(columns={0: "date", **cols})
    out["period"] = pd.PeriodIndex(out["date"].str.replace("M", "-"), freq="M")
    out = out.drop(columns="date").set_index("period").apply(pd.to_numeric, errors="coerce")
    return out


@lru_cache(maxsize=None)
def load_weo():
    """IMF WEO historical forecasts in long form:
    iso, target_year, vintage_year, season ('S' = April, 'F' = October), value."""
    w = pd.read_excel(C.RAW_DIR / "WEOhistorical.xlsx", sheet_name="ngdp_rpch")
    w = w.rename(columns={"ISOAlpha_3Code": "iso", "year": "target_year"})
    vcols = [c for c in w.columns if c[:1] in "SF" and c.endswith("ngdp_rpch")]
    long = w.melt(id_vars=["iso", "target_year"], value_vars=vcols,
                  var_name="vintage", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value", "iso"])
    long["season"] = long["vintage"].str[0]
    long["vintage_year"] = long["vintage"].str[1:5].astype(int)
    return long[["iso", "target_year", "vintage_year", "season", "value"]].reset_index(drop=True)


def weo_forecast(season="S"):
    """WEO forecast table: one row per (iso, vintage_year, target_year)."""
    w = load_weo()
    w = w[w["season"] == season]
    return w.pivot_table(index=["iso", "vintage_year", "target_year"], values="value").reset_index()


@lru_cache(maxsize=None)
def load_gep():
    return pd.read_csv(C.RAW_DIR / "gep_2026.csv")
