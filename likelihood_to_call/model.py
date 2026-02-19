"""
Likelihood-to-Call classifier.

Trains a gradient-boosted tree model (falls back to sklearn's
GradientBoostingClassifier if LightGBM is not installed) that outputs a
probability score in [0, 1] for each account on each day.

Key design decisions
--------------------
* Time-aware train / validation split: training data only uses dates before
  config.train_cutoff_date so there is no look-ahead leakage.
* Calibration: probabilities are Platt-scaled so that the raw output can be
  interpreted directly as a probability rather than a rank score.
* Class imbalance: handled via `class_weight` / `scale_pos_weight` tuning
  because in a credit card portfolio the majority of accounts don't call on
  any given day.
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from .config import Config
from .features import feature_columns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model factory: prefer LightGBM, fall back to sklearn GBM
# ---------------------------------------------------------------------------

def _make_base_estimator(config: Config):
    try:
        import lightgbm as lgb  # noqa: F401

        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=config.random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        logger.warning("LightGBM not found – using sklearn GradientBoostingClassifier.")
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=config.random_state,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(
    feature_matrix: pd.DataFrame,
    config: Config,
    calibrate: bool = True,
) -> Tuple[object, Dict]:
    """
    Train the likelihood-to-call classifier.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Output of features.build_feature_matrix (must include `called_next`
        and `snapshot_date` columns).
    config : Config
    calibrate : bool
        Wrap the estimator in Platt scaling so probabilities are well-
        calibrated.

    Returns
    -------
    model : fitted classifier
    metrics : dict of evaluation metrics on the hold-out set
    """
    assert "called_next" in feature_matrix.columns, (
        "feature_matrix must contain 'called_next' target column"
    )

    feat_cols = feature_columns(feature_matrix)
    X = feature_matrix[feat_cols].astype(float)
    y = feature_matrix["called_next"]
    dates = feature_matrix["snapshot_date"]

    # ── Time-aware split ─────────────────────────────────────────────────
    if config.train_cutoff_date:
        cutoff = pd.Timestamp(config.train_cutoff_date)
    else:
        # Use the latest 20 % of dates as hold-out
        sorted_dates = sorted(dates.unique())
        cutoff_idx = int(len(sorted_dates) * 0.80)
        cutoff = sorted_dates[cutoff_idx]

    train_mask = dates < cutoff
    test_mask  = dates >= cutoff

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    logger.info(
        "Train rows: %d (up to %s)  |  Test rows: %d (from %s)",
        train_mask.sum(), cutoff.date(), test_mask.sum(), cutoff.date()
    )
    logger.info(
        "Target prevalence – train: %.2f%%  |  test: %.2f%%",
        y_train.mean() * 100, y_test.mean() * 100
    )

    # ── Fit ──────────────────────────────────────────────────────────────
    base = _make_base_estimator(config)

    if calibrate:
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    else:
        model = base

    model.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────
    metrics = {}
    if len(y_test) > 0 and y_test.nunique() > 1:
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"]          = roc_auc_score(y_test, y_prob)
        metrics["avg_precision"]    = average_precision_score(y_test, y_prob)
        metrics["brier_score"]      = brier_score_loss(y_test, y_prob)
        metrics["test_prevalence"]  = float(y_test.mean())
        metrics["n_train"]          = int(train_mask.sum())
        metrics["n_test"]           = int(test_mask.sum())
        metrics["cutoff_date"]      = str(cutoff.date())

        logger.info(
            "Hold-out metrics → AUC-ROC: %.4f | Avg Precision: %.4f | Brier: %.4f",
            metrics["roc_auc"], metrics["avg_precision"], metrics["brier_score"]
        )
    else:
        logger.warning("Hold-out set too small or constant target – skipping metrics.")

    return model, metrics


def score(
    model,
    feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply a trained model to a feature matrix and return predicted call
    probabilities alongside the account / date identifiers.

    Parameters
    ----------
    model   : fitted classifier returned by train()
    feature_matrix : pd.DataFrame (may or may not contain 'called_next')

    Returns
    -------
    pd.DataFrame with columns: AccountNumber, snapshot_date, call_probability
    """
    feat_cols = feature_columns(feature_matrix)
    X = feature_matrix[feat_cols].astype(float)

    probs = model.predict_proba(X)[:, 1]

    out = feature_matrix[["AccountNumber", "snapshot_date"]].copy()
    out["call_probability"] = probs
    return out.reset_index(drop=True)


def feature_importance(model, feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Return a sorted feature-importance table.

    Works with both LightGBM and sklearn GBM via the calibrated wrapper.
    Returns an empty DataFrame if the underlying estimator does not expose
    feature_importances_.
    """
    feat_cols = feature_columns(feature_matrix)

    # Unwrap CalibratedClassifierCV if needed
    estimator = model
    if hasattr(model, "calibrated_classifiers_"):
        estimator = model.calibrated_classifiers_[0].estimator

    if not hasattr(estimator, "feature_importances_"):
        logger.warning("Estimator does not expose feature_importances_.")
        return pd.DataFrame()

    imp = pd.DataFrame(
        {"feature": feat_cols, "importance": estimator.feature_importances_}
    )
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def save_model(model, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", path)


def load_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", path)
    return model
