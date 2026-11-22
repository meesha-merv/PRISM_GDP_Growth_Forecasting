"""Baseline ladder, the gradient-boosting preview and the WEO benchmark.

Each model has fit(train) and predict(test), where train and test are rows
of the feature store. Every statistic a model needs (means, medians, clip
bounds, scaling) is learned from `train` only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as C
from . import data as D

try:                                   # the proposed model
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:                    # same algorithm family, ships with scikit-learn
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_LGB = False


class FoldClipper(BaseEstimator, TransformerMixin):
    """Clip each column at its 1st/99th percentile, learned on the training fold.
    Inflation and FDI have values in the thousands; this keeps one
    hyperinflation year from setting a linear coefficient."""
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, 1, axis=0)
        self.hi_ = np.nanpercentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def _clip_target(y):
    lo, hi = np.nanpercentile(y, [1, 99])
    return np.clip(y, lo, hi)


# ------------------------------------------------------------------ simple baselines
class Naive:
    """Next value = last value known at the cutoff."""
    def __init__(self, col="g_lag1_rt", name="Naive (real-time)"):
        self.col, self.name = col, name

    def fit(self, train):
        return self

    def predict(self, test):
        return test[self.col].to_numpy(dtype=float)


class CountryMean:
    """Expanding historical mean of the country's own growth."""
    name = "Country mean"

    def fit(self, train):
        self.mu_ = train.groupby("iso")["y"].mean()
        self.glob_ = train["y"].median()
        return self

    def predict(self, test):
        return test["iso"].map(self.mu_).fillna(self.glob_).to_numpy(dtype=float)


class PooledAR:
    """y_T = a + b1*g_lag1 + b2*g_lag2, one set of coefficients for all
    countries (global AR(2)). Target clipped at the training 1/99 percentiles."""
    name = "AR(2) pooled"
    cols = ["g_lag1_rt", "g_lag2_rt"]

    def _X(self, df):
        X = df[self.cols].copy()
        X["g_lag2_rt"] = X["g_lag2_rt"].fillna(X["g_lag1_rt"])
        return X.to_numpy(dtype=float)

    def fit(self, train):
        tr = train.dropna(subset=["g_lag1_rt"])
        self.m_ = LinearRegression().fit(self._X(tr), _clip_target(tr["y"].to_numpy()))
        return self

    def predict(self, test):
        X = self._X(test)
        out = np.full(len(test), np.nan)
        ok = ~np.isnan(X).any(axis=1)
        out[ok] = self.m_.predict(X[ok])
        return out


class FixedEffectsAR(PooledAR):
    """AR(2) with country fixed effects: each country reverts to its own
    training-period mean, at a speed shared by all countries."""
    name = "AR(2) + country FE"

    def fit(self, train):
        tr = train.dropna(subset=["g_lag1_rt"]).copy()
        y = _clip_target(tr["y"].to_numpy())
        self.mu_ = pd.Series(y, index=tr.index).groupby(tr["iso"]).mean()
        self.glob_ = float(np.mean(y))
        mu = tr["iso"].map(self.mu_).to_numpy()
        X = self._X(tr) - mu[:, None]
        self.m_ = LinearRegression(fit_intercept=False).fit(X, y - mu)
        return self

    def predict(self, test):
        mu = test["iso"].map(self.mu_).fillna(self.glob_).to_numpy(dtype=float)
        X = self._X(test) - mu[:, None]
        out = np.full(len(test), np.nan)
        ok = ~np.isnan(X).any(axis=1)
        out[ok] = mu[ok] + self.m_.predict(X[ok])
        return out


# ------------------------------------------------------------------ regularised linear
def _linear_pipeline(num_cols, cat_cols):
    num = Pipeline([
        ("clip", FoldClipper()),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat = OneHotEncoder(handle_unknown="ignore", min_frequency=1)
    pre = ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])
    return Pipeline([("pre", pre),
                     ("ridge", RidgeCV(alphas=np.logspace(-1, 3, 13)))])


class RidgeModel:
    """Regularised linear regression on lagged features.
    with_fe=True adds a one-hot dummy per country: penalised country fixed
    effects. Year fixed effects from the proposal are replaced by observed
    common factors (world growth, commodity prices), because a year effect
    for a future year cannot be estimated at forecast time."""
    def __init__(self, features, with_fe=True, name=None):
        self.features, self.with_fe = list(features), with_fe
        self.name = name or ("Ridge + country FE" if with_fe else "Ridge pooled")

    def fit(self, train):
        tr = train.dropna(subset=["g_lag1_rt"])
        cats = ["income_groups", "world_6region"] + (["iso"] if self.with_fe else [])
        self.cats_ = cats
        self.m_ = _linear_pipeline(self.features, cats)
        X = tr[self.features + cats].copy()
        X[cats] = X[cats].astype(str)
        self.m_.fit(X, _clip_target(tr["y"].to_numpy()))
        return self

    def predict(self, test):
        X = test[self.features + self.cats_].copy()
        X[self.cats_] = X[self.cats_].astype(str)
        out = self.m_.predict(X)
        out[test["g_lag1_rt"].isna().to_numpy()] = np.nan
        return out


# ------------------------------------------------------------------ gradient boosting preview
class GBM:
    """Global gradient-boosted trees on the same features (proposed model).
    Uses LightGBM when installed; otherwise scikit-learn's histogram GBM,
    which implements the same histogram-based leaf-wise-style boosting and
    handles missing values natively. Hyperparameters are deliberately
    conservative and NOT tuned yet (tuning is a Week-5 task)."""
    def __init__(self, features, name=None, seed=C.RANDOM_STATE):
        self.features = list(features)
        self.seed = seed
        self.name = name or ("LightGBM" if HAS_LGB else "GBM (HistGB)")

    def _X(self, df):
        X = df[self.features].copy()
        X["income_code"] = df["income_groups"].map(
            {"low_income": 0, "lower_middle_income": 1, "upper_middle_income": 2, "high_income": 3})
        X["region_code"] = df["world_6region"].astype("category").cat.codes
        return X

    def fit(self, train):
        tr = train.dropna(subset=["g_lag1_rt"])
        y = _clip_target(tr["y"].to_numpy())
        X = self._X(tr)
        if HAS_LGB:
            self.m_ = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                        min_child_samples=40, subsample=0.8, subsample_freq=1,
                                        colsample_bytree=0.7, reg_lambda=1.0,
                                        random_state=self.seed, verbose=-1)
            self.m_.fit(X, y, categorical_feature=["region_code"])
        else:
            cat_mask = [c == "region_code" for c in X.columns]
            self.m_ = HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40,
                l2_regularization=1.0, categorical_features=cat_mask,
                random_state=self.seed)
            self.m_.fit(X, y)
        return self

    def predict(self, test):
        out = self.m_.predict(self._X(test))
        out[test["g_lag1_rt"].isna().to_numpy()] = np.nan
        return out


# ------------------------------------------------------------------ institutional benchmark
class WEOBenchmark:
    """IMF WEO forecast for the target year, from the latest WEO round
    published by the cutoff (April round for an April cutoff)."""
    def __init__(self, name="IMF WEO"):
        self.name = name
        w = D.load_weo()
        self.w_ = w.set_index(["iso", "vintage_year", "season", "target_year"])["value"]
        self.w_ = self.w_[~self.w_.index.duplicated()]

    def fit(self, train):
        return self

    def predict(self, test):
        from .features import _latest_weo_vintage
        out = []
        for iso, t, ty, m in zip(test["iso"], test["year"], test["target_year"], test["cutoff_month"]):
            vy, s = _latest_weo_vintage(int(t), int(m))
            out.append(self.w_.get((iso, vy, s, int(ty)), np.nan))
        return np.array(out, dtype=float)


class WEOCorrection(RidgeModel):
    """Stacked correction: forecast = WEO + ridge model of the WEO error.
    Trained only on rows where the WEO forecast existed at the time."""
    def __init__(self, features, name="WEO + ridge correction"):
        super().__init__(features, with_fe=False, name=name)
        self.bench = WEOBenchmark()

    def fit(self, train):
        tr = train.copy()
        tr["weo"] = self.bench.predict(tr)
        tr = tr.dropna(subset=["weo", "g_lag1_rt"])
        tr = tr.assign(y=tr["y"] - tr["weo"])
        self.features_ = self.features + ["weo"]
        cats = ["income_groups", "world_6region"]
        self.cats_ = cats
        self.m_ = _linear_pipeline(self.features_, cats)
        X = tr[self.features_ + cats].copy()
        X[cats] = X[cats].astype(str)
        self.m_.fit(X, _clip_target(tr["y"].to_numpy()))
        return self

    def predict(self, test):
        te = test.copy()
        te["weo"] = self.bench.predict(te)
        X = te[self.features_ + self.cats_].copy()
        X[self.cats_] = X[self.cats_].astype(str)
        out = te["weo"].to_numpy() + self.m_.predict(X)
        out[te["weo"].isna().to_numpy()] = np.nan
        return out
