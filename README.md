# PRISM, Part 5: Baseline models and evaluation

Forecasting real GDP growth in South Asia with vintage-aware, mixed-frequency features, benchmarked against IMF WEO
forecasts. Chowdhury R Adnan · Meesha Mervyn

## Folder layout


```
└── 05_Modelling/                      
    ├── 05_Baseline_Models.ipynb       the workflow, step by step, with outputs
    ├── run_experiments.py             regenerates every table and figure
    ├── notebook_cells.py              source of the notebook (plain Python, cell markers)
    ├── requirements.txt
    ├── prism/
    │   ├── config.py                  paths, countries, fiscal years, publication lags
    │   ├── data.py                    loaders for WDI, GEM, Pink Sheet, WEO, GEP (no filling)
    │   ├── reconcile.py               reporting-period checks and effective horizons
    │   ├── features.py                vintage-aware feature store
    │   ├── models.py                  baseline ladder, GBM, WEO benchmark, stacked correction
    │   ├── evaluate.py                rolling origin, scores, Diebold-Mariano, conformal intervals
    │   └── experiments.py             the experiments and figures in the progress report
    └── outputs/
        ├── tables/*.csv               every number quoted in the report
        ├── figures/*.png
        └── cache/                     feature stores and predictions (safe to delete)
```

## How to run

```bash
cd 05_Modelling
pip install -r requirements.txt        # lightgbm is optional
python run_experiments.py              
python run_experiments.py --fresh      # rebuild everything from the raw files
```

Or open `05_Baseline_Models.ipynb` and run all cells.

**LightGBM.** `models.GBM` uses LightGBM when it is installed. Otherwise it falls back to scikit-learn's
`HistGradientBoostingRegressor` (same histogram-based boosting, native missing-value handling). The saved results were
produced with the fallback and are labelled `GBM (HistGB)`. Install LightGBM and delete `outputs/cache/` to rerun with it.

## Workflow

### 1. Reconcile reporting periods: `reconcile.py`
* Compares the IMF's own history of each year with WDI years *t*−1, *t*, *t*+1 for every South Asian country.
  All eight match at shift 0, so the WEO comparison is like-for-like on labels.
* Runs the same check on the June-2026 GEP file (India is ambiguous, likely the 2026 GDP rebasing).
* Computes the **effective horizon**: months left in each country's reporting year at the cutoff
  (April cutoff: Pakistan 2, Sri Lanka 8, India 11).

### 2. Build the feature store: `features.build_store(cutoff_month, horizon)`
One row per (country, origin year *t*), holding only what was public at the end of month *m* of year *t*:

| Block | Rule |
|---|---|
| Growth history | real-time *t*−1, *t*−2 growth from the latest WEO round at the cutoff (fallback: WDI) |
| Annual WDI | year *t*−1 only from July (WDI release), else *t*−2; forward fill within country, max 2 years |
| Monthly GEM | 3-month y/y up to the last published month (lags 0–2 months); fiscal-year-to-date growth of IP, exports, imports |
| Pink Sheet | 3-month y/y of energy, non-energy, food, metals; lag 1 month |
| Global cycle | median growth of all countries in the last known year |

No cross-sectional (income-group) imputation. Remaining gaps stay `NaN`; linear models fill them with the
**training-fold** median and a missing flag; the GBM takes `NaN` directly. Series above 90% missing are dropped.

### 3. Model ladder: `models.py`
Naive (real-time) · Naive (revised data) · Country mean · AR(2) pooled · AR(2) + country FE · Ridge + FE (annual only) ·
Ridge + country FE · Ridge pooled · GBM · **IMF WEO** (benchmark) · WEO + ridge correction (stacked).

### 4. Rolling-origin evaluation: `evaluate.rolling_origin`
For each origin *t* = 2000 … 2024: train on every country-year whose outcome was known at the cutoff
(target year ≤ *t*−1, origin ≥ 1992), refit every model, predict year *t*+*h* for all countries. South Asia is scored;
the rest of the world is a check. The IMF head-to-head window is 2000–2022 (last WEO vintage: October 2022).

### 5. Scores and tests: `evaluate.score_table`, `evaluate.dm_test`
RMSE, MAE, MAPE (|growth| > 1 pp only), out-of-sample R², RMSE relative to the naive forecast, on a common sample.
Diebold–Mariano (HLN-corrected) against the WEO and the naive forecast. Reported with and without 2020–21, per country,
by effective horizon, and against two "truths" (today's WDI, the IMF's first published estimate).

### 6. Experiments: `experiments.py`
* Feature ablation: growth only → + annual → + monthly → + Pink Sheet → revised growth lags.
* Conformal 80% intervals: global, country-scaled, adaptive (ACI); coverage overall, outside and inside 2020–21.
* Monthly refresh: the same current-year target at every month-end cutoff, January to December.

## Final stage

```bash
python run_experiments.py      # the progress-report ladder (needed first; reproduces all 30 tables)
python run_final.py            # tuning, RQ1 arms, importance, RQ2 intervals, monthly refresh -> outputs/tables/final_*.csv
python train_final.py          # final model + deployment bundle 
```

New modules:
* `prism/final.py` – quarterly GDP block, final feature set (Pink Sheet dropped), GBM tuned every five origins on the
  three preceding origins (inner time splits), RQ1 arms A/B/C0/C, permutation importance, Diebold-Mariano tests on
  squared and absolute loss, crisis-aware conformal intervals and the interval score.
* `prism/live.py` – feature store for the live 2026 origin (last year's growth from the GEP June 2026 estimate).

Final model ("C. PRISM final"): growth history (real-time) + annual WDI + monthly GEM + quarterly GDP, scikit-learn
histogram gradient boosting with tuned hyperparameters. South Asia, April cutoff, h = 0, 2000-2022: RMSE 6.37
(IMF WEO 4.98, AR(2) 6.57, naive 8.31); excluding 2020-21: 4.61 (IMF 3.95). 
