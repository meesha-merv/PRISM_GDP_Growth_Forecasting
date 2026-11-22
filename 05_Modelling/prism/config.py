"""Project settings: paths, country lists, fiscal years and publication lags.

Every assumption that decides what the model is allowed to see sits in this
file, so it can be checked and changed in one place.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
# The code lives in <project>/05_Modelling/prism. The data folders sit in
# <project>/wdi-dataset/wdi and <project>/raw-datafiles/raw, exactly as they
# are unzipped from the submission archives.
PKG_DIR = Path(__file__).resolve().parent
MODEL_DIR = PKG_DIR.parent
PROJECT_DIR = MODEL_DIR.parent


def _find(*candidates):
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError("None of these folders exist: " + ", ".join(map(str, candidates)))


WDI_DIR = _find(PROJECT_DIR / "wdi-dataset" / "wdi", PROJECT_DIR / "data" / "wdi")
RAW_DIR = _find(PROJECT_DIR / "raw-datafiles" / "raw", PROJECT_DIR / "data" / "raw")
OUT_DIR = MODEL_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
for d in (OUT_DIR, FIG_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- countries
SA = ["AFG", "BGD", "BTN", "IND", "LKA", "MDV", "NPL", "PAK"]
SA_NAMES = {
    "AFG": "Afghanistan", "BGD": "Bangladesh", "BTN": "Bhutan", "IND": "India",
    "LKA": "Sri Lanka", "MDV": "Maldives", "NPL": "Nepal", "PAK": "Pakistan",
}

# ---------------------------------------------------------------- fiscal years
# National-accounts reporting period for each South Asian economy.
#   fy_end_month : month in which the reporting year ends (12 = calendar year)
#   label        : how the year t in WDI maps to the fiscal year
#                  "end"   -> year t is the fiscal year that ENDS in t
#                  "start" -> year t is the fiscal year that STARTS in t
# These are the documented conventions (World Bank / IMF country notes). The
# reconcile module checks them against the data rather than trusting them.
FISCAL_YEAR = {
    "AFG": {"fy_end_month": 12, "label": "end",   "note": "solar year, close to calendar since 2012"},
    "BGD": {"fy_end_month": 6,  "label": "end",   "note": "July-June"},
    "BTN": {"fy_end_month": 6,  "label": "end",   "note": "July-June"},
    "IND": {"fy_end_month": 3,  "label": "start", "note": "April-March, year t = FY t/t+1"},
    "LKA": {"fy_end_month": 12, "label": "end",   "note": "calendar year"},
    "MDV": {"fy_end_month": 12, "label": "end",   "note": "calendar year"},
    "NPL": {"fy_end_month": 7,  "label": "end",   "note": "mid-July to mid-July"},
    "PAK": {"fy_end_month": 6,  "label": "end",   "note": "July-June"},
}


def fy_window(iso, year):
    """Return (start, end) of the reporting year `year` for country `iso`
    as pandas-style (year, month) tuples. Non-South-Asian countries are
    treated as calendar-year reporters (a stated simplification)."""
    fy = FISCAL_YEAR.get(iso, {"fy_end_month": 12, "label": "end"})
    m = fy["fy_end_month"]
    end_year = year if fy["label"] == "end" else year + 1
    if m == 12:
        end_year = year
    end = (end_year, m)
    # start is 11 months before end
    sy, sm = end_year, m - 11
    if sm <= 0:
        sm += 12
        sy -= 1
    return (sy, sm), end


def months_to_fy_end(iso, year, cutoff_month, origin_year):
    """Months between the information cutoff (end of `cutoff_month` in
    `origin_year`) and the end of reporting year `year`. This is the
    *effective* horizon: an April cutoff is 2 months before the end of
    Pakistan's year but 11 months before the end of India's."""
    _, (ey, em) = fy_window(iso, year)
    return (ey - origin_year) * 12 + (em - cutoff_month)


# ---------------------------------------------------------------- publication lags
# Annual WDI indicators for year t-1 are released in the July update of
# year t. Before that only t-2 is public. GDP growth for t-1 is the exception:
# a preliminary national estimate is public by the spring of year t (it is the
# "last actual" printed in the April WEO), so it is available from March.
ANNUAL_RELEASE_MONTH = 7
GDP_PRELIM_MONTH = 3
MAX_ANNUAL_STALENESS = 2   # carry an annual value forward at most 2 years

# Monthly GEM series: months between the end of the reference month and the
# date the value is public. Conservative, rounded up.
MONTHLY_LAG = {
    "DPANUSSPB": 0,      # exchange rate, LCU per USD (market data)
    "REER": 1,           # real effective exchange rate
    "CPTOTSAXN": 1,      # CPI level, seasonally adjusted
    "CPTOTSAXNZGY": 1,   # CPI inflation, y/y
    "IPTOTSAKD": 2,      # industrial production
    "DXGSRMRCHSACD": 2,  # merchandise exports, USD
    "DMGSRMRCHSACD": 2,  # merchandise imports, USD
    "TOTRESV": 2,        # total reserves
    "IMPCOV": 2,         # reserves in months of imports
}
PINK_LAG = 1             # Pink Sheet is published in the first days of the next month

# ---------------------------------------------------------------- evaluation
BENCH_CUTOFFS = {"WEO_spring": 4, "GEP_june": 6}
TEST_YEARS = list(range(2000, 2025))   # forecast origins for rolling evaluation
WEO_LAST_VINTAGE = 2022                # last vintage in WEOhistorical.xlsx
CRISIS_YEARS = [2020, 2021]
ALPHA = 0.20                           # 80% prediction intervals
RANDOM_STATE = 42
