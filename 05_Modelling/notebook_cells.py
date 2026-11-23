# %% [markdown]
# # GDP Growth Forecasting Baseline Models and Evaluation
#
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
# 
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

#
# %%
R.gep_alignment().round(2)


# %%
R.effective_horizons(4)


# %%
store = E.load_store(4, 0)
groups = F.feature_groups(store)
print(store.shape)
{k: len(v) for k, v in groups.items()}

# %%
# What a forecaster knew about Sri Lanka at end-April of each year
cols = ['iso', 'year', 'target_year', 'y', 'g_lag1', 'g_lag1_rt', 'exports_m_fytd', 'pink_energy_yoy3', 'remittances_a', 'months_ahead']
store.query("iso == 'LKA' and 2018 <= year <= 2023")[cols].round(2)


# %%
sa = store[(store.is_sa == 1) & store.year.between(2000, 2024)]
miss = sa[groups['annual'] + groups['monthly']].isna().mean().sort_values(ascending=False)
miss.head(12).round(2).to_frame('share missing')


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
# * **global**: one half-width for everyone;
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

