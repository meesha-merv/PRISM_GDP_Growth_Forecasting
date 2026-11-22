"""Reconcile reporting periods across WDI, IMF WEO and World Bank GEP.

The feedback point: India reports April-March, Pakistan, Bangladesh and
Bhutan July-June, Nepal mid-July to mid-July, Sri Lanka and Maldives the
calendar year. If two sources label the same fiscal year with different
calendar years, a 'forecast error' is partly a labelling error.

We do not trust the documentation alone. For every country we line up the
WEO history (the value of year y printed two years later, when y is already
history) against WDI year y+k for k = -1, 0, +1 and keep the shift with the
smallest gap. A shift of 0 with a small gap confirms the labels agree.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import data as D


def weo_history(lag=2, season="F"):
    """WEO 'actual' for target year y as printed in vintage y+lag."""
    w = D.load_weo()
    w = w[(w["season"] == season) & (w["vintage_year"] == w["target_year"] + lag)]
    return w.set_index(["iso", "target_year"])["value"]


def label_alignment(countries=C.SA, shifts=(-1, 0, 1)):
    wdi = D.load_wdi().set_index(["iso", "year"])["gdp_growth"]
    hist = weo_history()
    rows = []
    for iso in countries:
        h = hist.xs(iso, level="iso") if iso in hist.index.get_level_values(0) else pd.Series(dtype=float)
        if iso in wdi.index.get_level_values(0):
            w = wdi.xs(iso, level="iso")
        else:
            w = pd.Series(dtype=float)
        res = {"iso": iso, "country": C.SA_NAMES.get(iso, iso),
               "fiscal_year": C.FISCAL_YEAR.get(iso, {}).get("note", "calendar")}
        for k in shifts:
            both = pd.concat([h, w.shift(-k)], axis=1, keys=["weo", "wdi"]).dropna()
            res[f"MAE_shift_{k:+d}"] = (both["weo"] - both["wdi"]).abs().mean()
            res[f"n_shift_{k:+d}"] = len(both)
        best = min(shifts, key=lambda k: res[f"MAE_shift_{k:+d}"])
        res["best_shift"] = best
        rows.append(res)
    return pd.DataFrame(rows)


def gep_alignment(countries=C.SA, shifts=(-1, 0, 1)):
    """Same check for the June-2026 GEP file, whose 2023-2025 columns are
    estimates of years that have already happened."""
    gep = D.load_gep().set_index(["iso", "year"])["gep_jun2026_growth"]
    wdi = D.load_wdi().set_index(["iso", "year"])["gdp_growth"]
    rows = []
    for iso in countries:
        g = gep.xs(iso, level="iso")
        w = wdi.xs(iso, level="iso")
        res = {"iso": iso}
        for k in shifts:
            both = pd.concat([g, w.shift(-k)], axis=1, keys=["gep", "wdi"]).dropna()
            both = both[both.index <= 2024]
            res[f"gap_shift_{k:+d}"] = (both["gep"] - both["wdi"]).abs().mean()
            res[f"years_shift_{k:+d}"] = ",".join(map(str, both.index))
        rows.append(res)
    return pd.DataFrame(rows)


def effective_horizons(cutoff_month=4, horizons=(0, 1), countries=C.SA):
    """Months from the cutoff to the end of the target reporting year."""
    rows = []
    for iso in countries:
        start, end = C.fy_window(iso, 2020)
        row = {"iso": iso, "country": C.SA_NAMES[iso],
               "reporting_year_2020": f"{start[0]}-{start[1]:02d} to {end[0]}-{end[1]:02d}"}
        for h in horizons:
            row[f"months_ahead_h{h}"] = C.months_to_fy_end(iso, 2020 + h, cutoff_month, 2020)
        rows.append(row)
    return pd.DataFrame(rows)
