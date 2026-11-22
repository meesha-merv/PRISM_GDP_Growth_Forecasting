# %% [markdown]
# # GDP Growth Forecasting, Part 5: Baseline Models and Evaluation
#
# **Project:** Forecasting real GDP growth in South Asia · Chowdhury R Adnan (st127357) & Meesha Mervyn (st127378)
#
# Parts 1–4 (`01_Acquire_Data` … `04_Bivariate_Analysis`) explored the WDI panel. This part builds the forecasting
# system and scores the baseline ladder. It answers the three weaknesses raised at the proposal presentation:
#
# | Feedback | What this notebook does |
# |---|---|
# | **B1.** Imputation can leak information not available at the forecast date | A vintage-aware feature store. No income-group median fill; values carried forward only inside a series' publication lag; remaining gaps filled *inside each training fold* with a missing flag. Real-time (as-published) values of last year's growth replace revised WDI values. |
# | **B2.** IMF / World Bank comparison ignores country reporting periods | Step 1 checks the year labels of WEO, GEP and WDI country by country, and computes each country's *effective horizon* (months left in its reporting year at the cutoff). Monthly data are also aggregated over each country's own fiscal year. |
# | **B3.** Experiments too broad, no prioritised reproducible baseline | One fixed baseline ladder, one evaluation protocol, one command (`python run_experiments.py`) that regenerates every table and figure. Extensions are ranked P1–P3 in the progress report. |
#
# ### Workflow
#
# ```
#  raw files ──► 1. reconcile reporting periods ──► 2. feature store at cutoff (month m, horizon h)
#                                                              │
#                3. model ladder (naive → AR → FE → ridge → GBM, + WEO benchmark)
#                                                              │
#                4. rolling-origin loop: for each year t, refit on outcomes known at the cutoff, predict t+h
#                                                              │
#     5. scores & paired tests ◄─┬─► 6. feature ablation ─┬─► 7. conformal intervals ─► 8. monthly refresh
# ```
#
# All code lives in the `prism/` package next to this notebook. The notebook only calls it, so the
# command-line runner and the notebook cannot disagree.
#
# > **Runtime.** The first full run builds 15 feature stores and takes about 20 minutes. Results are cached in
# > `outputs/cache/`; later runs take seconds. Delete that folder (or run `python run_experiments.py --fresh`) to rebuild.

# %% [markdown]
# ## 0. Setup

# %%
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)

from prism import config as C, data as D, features as F, models as M
from prism import evaluate as E, reconcile as R, experiments as X

try:
    from IPython.display import Image
except ImportError:               # lets the notebook run as a plain script too
    Image = lambda filename: Path(filename)

GBM = 'LightGBM' if M.HAS_LGB else 'GBM (HistGB)'
print('WDI folder :', C.WDI_DIR)
print('Raw folder :', C.RAW_DIR)
print('Gradient boosting backend:', GBM)

# %% [markdown]
# **About the gradient-boosting backend.** The proposed model is LightGBM. `models.GBM` uses LightGBM whenever
# it is installed (`pip install lightgbm`). If it is not, it falls back to scikit-learn's
# `HistGradientBoostingRegressor`, which uses the same histogram-based boosting and handles missing values
# natively. The results saved in this notebook were produced with the fallback; the LightGBM run is a Week-5 task.
# At this stage the GBM is a **preview** with untuned, conservative settings; the point of Part 5 is the baselines.

# %% [markdown]
# ## 1. Reconcile reporting periods (feedback B2)
#
# South Asian countries do not all report on the calendar year:
#
# | Country | Reporting year | WDI year *t* means |
# |---|---|---|
# | India | April–March | FY *t*/*t*+1 |
# | Bangladesh, Bhutan, Pakistan | July–June | FY *t*−1/*t* |
# | Nepal | mid-July–mid-July | FY *t*−1/*t* |
# | Sri Lanka, Maldives, Afghanistan | calendar (Afghanistan: solar year) | calendar *t* |
#
# Two things can go wrong. First, two sources may put the same fiscal year under different labels.
# Second, even with matching labels, an April forecast of "year *t*" is a very different task for each country.
#
# **Label check.** We take the IMF's own history of each year (the value printed two WEO rounds later, once the
# year is over) and compare it with WDI year *t*−1, *t* and *t*+1. The shift with the smallest gap is the label match.

# %%
align = R.label_alignment()
align[['country', 'fiscal_year', 'MAE_shift_-1', 'MAE_shift_+0', 'MAE_shift_+1', 'best_shift']].round(2)

# %% [markdown]
# Every country matches at shift 0, with a gap 1.6 to 2.7 times smaller than at ±1. **WEO and WDI use the same year
# labels**, so the IMF comparison is like-for-like on labels. The gap at shift 0 is not zero, though: it is the
# **revision gap** between what the IMF printed and what WDI shows today (about 0.5 pp for Bangladesh, 0.9 pp for India,
# 2.6 pp for Maldives). That is direct evidence that the choice of "truth" vintage matters (see Step 5c).
#
# The GEP file holds only the June 2026 vintage. Its 2023–2024 columns are already history, so we run the same check:

# %%
R.gep_alignment().round(2)

# %% [markdown]
# GEP matches WDI at shift 0 for seven countries (gaps of 0.02–0.33 pp). **India is ambiguous**: shift 0 and shift +1
# give similar gaps, most likely because India rebased its national accounts in 2026, so GEP and the older WDI
# snapshot are on different base years. This is flagged as an open item. It matters only for the forward-looking
# 2026–27 comparison, because **the file has no historical GEP vintages, so the World Bank cannot be scored
# historically yet** (see the progress report, challenges).
#
# **Effective horizon.** At an end-April cutoff, how many months of each country's target year are still ahead?

# %%
R.effective_horizons(4)

# %% [markdown]
# At the April WEO round, a "current-year" forecast is a near-nowcast for Pakistan, Bangladesh, Bhutan and Nepal
# (2–3 months left), an 8-month-ahead forecast for Sri Lanka, Maldives and Afghanistan, and an 11-month-ahead forecast
# for India. Pooling them as one "h = 0" hides this, so Step 5d reports errors by effective horizon. The feature
# store also adds **fiscal-year-to-date** monthly features that average only the months of the target reporting year.

# %% [markdown]
# ## 2. The vintage-aware feature store (feedback B1)
#
# `features.build_store(cutoff_month, horizon)` returns one row per (country, origin year *t*). Each row holds
# only what was public at the end of month *m* of year *t*, and the realised growth of year *t*+*h*.
#
# | Block | Source | Rule at cutoff (year *t*, month *m*) |
# |---|---|---|
# | Growth history | WEO vintages | Real-time growth of *t*−1 and *t*−2 **as printed in the latest WEO round** (Oct *t*−1, Apr *t* or Oct *t*). Falls back to WDI only where WEO has no number. |
# | Annual structure | WDI | Year *t*−1 values only from July (the WDI release); before that, *t*−2. Carried forward within the country for at most 2 years. |
# | Monthly country signals | GEM | 3-month average, year-on-year, up to the last month published (lags of 0–2 months by series). Plus fiscal-year-to-date growth of IP, exports and imports for the target year. |
# | Global shocks | Pink Sheet | 3-month y/y change in energy, non-energy, food and metals indices, lag 1 month. |
# | Global cycle | WDI | Median growth of all countries in the last known year. |
#
# **What changed from Part 2.** Part 2 filled gaps with a 2-year forward fill and then the **median of the same
# income group**. That median is taken across countries *in the same year*, including years after the forecast date,
# and it is computed on the whole sample. Here there is no cross-sectional fill. Remaining gaps stay `NaN`; the linear
# models fill them with the median **of the training fold only**, plus a 0/1 missing flag. The tree model takes `NaN` directly.

# %%
store = E.load_store(4, 0)
groups = F.feature_groups(store)
print(store.shape)
{k: len(v) for k, v in groups.items()}

# %%
# What a forecaster knew about Sri Lanka at end-April of each year
cols = ['iso', 'year', 'target_year', 'y', 'g_lag1', 'g_lag1_rt', 'exports_m_fytd', 'pink_energy_yoy3', 'remittances_a', 'months_ahead']
store.query("iso == 'LKA' and 2018 <= year <= 2023")[cols].round(2)

# %% [markdown]
# `g_lag1` is last year's growth as WDI shows it **today**; `g_lag1_rt` is the number the IMF printed in April of that
# year. For Sri Lanka in April 2019, the 2018 figure was 3.0% in real time but 2.3% after revision. A model trained
# on `g_lag1` learns from numbers nobody had. (From 2023 the WEO file ends, so `g_lag1_rt` falls back to WDI.)
#
# How much of the South Asian feature matrix is missing under the new rules (rows used for evaluation, 2000–2024)?

# %%
sa = store[(store.is_sa == 1) & store.year.between(2000, 2024)]
miss = sa[groups['annual'] + groups['monthly']].isna().mean().sort_values(ascending=False)
miss.head(12).round(2).to_frame('share missing')

# %% [markdown]
# Industrial production, the REER and the monthly trade series are missing for several South Asian economies (GEM
# does not cover Afghanistan, Bhutan or Nepal for most of them). Under Part 2's method these cells would have been
# filled with the income-group median; here they stay empty and the model is told they are empty.

# %% [markdown]
# ## 3. The model ladder
#
# | # | Model | What it uses | Why it is on the ladder |
# |---|---|---|---|
# | 1 | **Naive (real-time)** | last year's growth as printed at the time | the reference: all skill is measured against it |
# | 1b | Naive (revised data) | last year's growth as revised later | shows the cost of using revised data |
# | 2 | Country mean | the country's own past average | the "long-run trend" forecast |
# | 3 | **AR(2) pooled** | two real-time growth lags, one coefficient set for all countries | the standard time-series baseline |
# | 4 | AR(2) + country FE | same, each country reverts to its own mean | proposal's fixed-effects baseline |
# | 5 | Ridge + FE (annual only) | growth lags + annual WDI + country dummies | regularised linear FE, annual data only |
# | 6 | Ridge + country FE | all features + country dummies | adds monthly and global data |
# | 7 | **Ridge pooled** | all features, income and region dummies | regularised linear without country dummies |
# | 8 | **GBM** (preview) | all features, NaN kept | the proposed model family |
# | B | **IMF WEO** | WEO forecast from the latest round at the cutoff | the institutional benchmark (held out, never a feature except in 9) |
# | 9 | WEO + ridge correction | WEO plus a ridge model of WEO's past error | the proposal's stacked correction |
#
# **A correction to the proposal.** The proposal listed *country and year* fixed effects. Year effects cannot be used
# to forecast: the effect of a future year is unknown at the forecast date. They are replaced with observed common
# factors (world growth, commodity prices), which a forecaster does have.
#
# Linear models and the GBM are trained on the target clipped at the training fold's 1st/99th percentiles (the
# winsorising from Part 3). Scores always use the **unclipped** realised value.

# %%
[m.name for m in X.model_ladder(store)]

# %% [markdown]
# ## 4. Rolling-origin evaluation
#
# For every origin year *t* from 2000 to 2024:
#
# 1. **Train** on every country-year (all 208 economies, South Asia's own past included) whose outcome was public at
#    the cutoff: target year ≤ *t*−1, origin year ≥ 1992 (the start of the monthly data).
# 2. **Refit** every model from scratch on that training set.
# 3. **Predict** year *t*+*h* for every country.
#
# The window expands by one year at each step. Nothing from year *t* or later reaches the model.
# South Asia is the evaluation set; the rest of the world is also scored, as a check.
#
# The IMF benchmark exists for target years up to 2022 (the last vintage in `WEOhistorical.xlsx` is October 2022),
# so the head-to-head window is **2000–2022**: 181 South Asian country-years at *h* = 0 and 174 at *h* = 1.

# %%
preds = {tag: X.run_ladder(m, h) for tag, (m, h) in
         {'m04_h0': (4, 0), 'm04_h1': (4, 1), 'm06_h0': (6, 0), 'm06_h1': (6, 1)}.items()}
p = preds['m04_h0']
print(p.shape)
p.head()

# %% [markdown]
# ## 5. Results
#
# ### 5a. April cutoff, current year (*h* = 0), South Asia, 2000–2022
# All models are scored on the same country-years. `RMSE_vs_naive` below 1 means better than the naive forecast.
# MAPE is computed only where |growth| > 1 pp, because it explodes near zero growth.

# %%
E.score_table(p, years=X.WEO_WINDOW).round(2)

# %%
E.score_table(p, years=X.WEO_WINDOW, exclude_crisis=True).round(2)

# %%
Image(filename=str(C.FIG_DIR / 'fig1_baseline_ladder.png'))

# %% [markdown]
# ### 5b. April cutoff, next year (*h* = 1)

# %%
E.score_table(preds['m04_h1'], years=X.WEO_WINDOW).round(2)

# %%
E.score_table(preds['m04_h1'], years=X.WEO_WINDOW, exclude_crisis=True).round(2)

# %% [markdown]
# ### 5c. Paired tests against the IMF and against the naive forecast
# Diebold–Mariano test on squared errors with the Harvey–Leybourne–Newbold correction, pooled over South Asian
# country-years. A positive `mean_loss_diff` means the model is *worse* than the benchmark.

# %%
dm0 = pd.read_csv(C.TAB_DIR / 'm04_h0_dm_tests.csv', index_col=0)
dm1 = pd.read_csv(C.TAB_DIR / 'm04_h1_dm_tests.csv', index_col=0)
pd.concat({'h=0': dm0[dm0.bench == 'IMF WEO'].set_index('model'),
           'h=1': dm1[dm1.bench == 'IMF WEO'].set_index('model')}, axis=1)[
    [('h=0', 'mean_loss_diff'), ('h=0', 'p_value'), ('h=1', 'mean_loss_diff'), ('h=1', 'p_value')]].round(3)

# %% [markdown]
# **Which truth?** Scored against the IMF's first published estimate (a year later) instead of today's WDI value:

# %%
pd.read_csv(C.TAB_DIR / 'm04_h0_scores_vs_first_release.csv', index_col=0)

# %% [markdown]
# ### 5d. Where the errors are: by country and by effective horizon

# %%
pd.read_csv(C.TAB_DIR / 'm04_h0_rmse_by_country.csv', index_col=0)

# %%
pd.read_csv(C.TAB_DIR / 'm04_h0_rmse_by_effective_horizon.csv', index_col=0)

# %%
Image(filename=str(C.FIG_DIR / 'fig5_effective_horizon.png'))

# %%
Image(filename=str(C.FIG_DIR / 'fig3_country_forecasts.png'))

# %% [markdown]
# ### 5e. June cutoff (the GEP round)
# The World Bank cannot be scored historically with the files we have, so at June the benchmark is still the April WEO
# (the latest IMF round public in June). Our models gain two more months of monthly data.

# %%
E.score_table(preds['m06_h0'], years=X.WEO_WINDOW).round(2)

# %% [markdown]
# ## 6. Feature ablation: does mixed-frequency data help? (towards research question 1)
# Ridge pooled and the GBM refit with growing feature sets, April cutoff, *h* = 0, test years 2000–2024. The last row
# swaps the real-time growth lags for revised WDI values, to measure the vintage effect.

# %%
X.ablation(4, 0)

# %% [markdown]
# ## 7. Prediction intervals (towards research question 2)
# Split-conformal 80% intervals, built only from earlier out-of-sample residuals (last 10 origins, all countries):
#
# * **global**: one half-width for everyone (the proposal's design);
# * **scaled**: residuals divided by each country's own past residual spread, so volatile economies get wider bands;
# * **adaptive**: scaled, plus adaptive conformal inference: each country's miscoverage level moves after every
#   hit or miss, once that outcome is public.

# %%
cov = pd.read_csv(C.TAB_DIR / 'm04_h0_interval_coverage.csv', index_col=0)
cov[cov.model == GBM].pivot(index='sample', columns='method', values=['coverage_%', 'mean_width_pp']).round(1)

# %%
Image(filename=str(C.FIG_DIR / 'fig4_interval_coverage.png'))

# %% [markdown]
# ## 8. Monthly refresh
# The same current-year target re-estimated at every month-end. Each line uses one common sample per month
# (South Asia, 2000–2022). The IMF line steps at its April and October rounds; January–March use the previous October round.

# %%
pd.read_csv(C.TAB_DIR / 'refresh_h0_rmse_by_cutoff.csv', index_col=0)[
    ['Naive (real-time) [2000-22]', 'AR(2) pooled [2000-22]', 'Ridge pooled [2000-22]', f'{GBM} [2000-22]', 'IMF WEO [2000-22]']]

# %%
Image(filename=str(C.FIG_DIR / 'fig2_monthly_refresh.png'))

# %% [markdown]
# ## 9. What the findings suggest
#
# 1. **Every model beats the naive forecast, and the IMF beats every model.** At the April cutoff (*h* = 0) the best
#    feature model (GBM preview, RMSE 6.35 pp) is 23% better than the real-time naive forecast (8.25), but the WEO is
#    better still (4.86). The gap is statistically significant at *h* = 0 (DM p ≈ 0.03 for the GBM), not at *h* = 1
#    (p ≈ 0.14), where, once 2020–21 are excluded, the AR(2) and GBM are 13–14% behind the IMF (4.33 and 4.35 vs 3.83).
# 2. **Simple beats complex among the baselines.** Pooled AR(2) and pooled ridge match or beat the fixed-effects
#    versions. Country dummies add noise: 60 years per country are too few to estimate them well. This confirms the
#    global-panel design.
# 3. **Revised data flatter.** The naive forecast built from revised data is 12% worse for South Asia than the real-time
#    one at *h* = 0; in the ablation, revised lags look *better* for the rest of the world (5.61 vs 5.67) but *worse*
#    for South Asia (6.38 vs 6.25). Revised history is not a free gain, and its look-ahead can mislead evaluation.
# 4. **Monthly data help, a little, and keep helping through the year.** Adding GEM monthly features lowers GBM RMSE
#    from 6.29 to 6.24 (South Asia) and 5.74 to 5.65 (rest of world). Through the year, GBM error falls from 6.43 (April)
#    to 5.84 (December). The WEO falls faster, because it uses quarterly national accounts and country desk knowledge we do not have.
# 5. **The IMF's lead is largest at the long effective horizon.** For India (11 months left at April) the IMF's RMSE is
#    2.57 against 3.71; for the July–June reporters (2–3 months left) the gap is smaller (2.49 vs 2.99).
# 6. **Intervals are calibrated in normal years and fail in crises.** Global conformal intervals cover 83% of South Asian
#    outcomes outside 2020–21, but only 38% in 2020–21. Scaling and adaptive updating did not fix the crisis case in this first try.
# 7. **Maldives and Afghanistan dominate squared error** (RMSE 13.4 and 9.5 for the GBM). Results are always reported with
#    and without 2020–21, and per country.
#
# Next steps are in the progress report: switch to LightGBM and tune it (P1), add a quarterly-GDP/nowcast block and
# fiscal-year-aware targets (P1), crisis-aware intervals (P2), GEP vintages (P2).
