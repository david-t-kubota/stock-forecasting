"""
Feature engineering for the Likelihood-to-Call model.

For every (AccountNumber, snapshot_date) pair we compute backward-looking
features drawn from the Ledger and a forward-looking binary target:
    did the account place an Inbound Call in the next `prediction_horizon_days`?

Design notes
------------
* All look-back computations use strictly historical data (no leakage).
* The feature matrix is built by iterating over a list of snapshot dates;
  in production this list is typically [today] so a single day's scores are
  produced efficiently.
* "Active" accounts on a snapshot date are those with at least one Ledger
  event in the prior `max_days_since` days (configurable).
"""
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_matrix(
    ledger: pd.DataFrame,
    config: Config,
    snapshot_dates: Optional[List[pd.Timestamp]] = None,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Build a feature matrix from the raw Ledger DataFrame.

    Parameters
    ----------
    ledger : pd.DataFrame
        Output of data_loader.load_ledger / generate_synthetic_ledger.
    config : Config
    snapshot_dates : list of Timestamps, optional
        Dates on which to score accounts.  Defaults to every date in the
        ledger (useful for training).  For scoring pass [pd.Timestamp.today()].
    include_target : bool
        If True, adds the binary `called_next_Nd` column.  Set to False when
        scoring future dates where the target is unknown.

    Returns
    -------
    pd.DataFrame with one row per (AccountNumber, snapshot_date).
    """
    ledger = ledger.copy()
    ledger["EffectiveDate"] = pd.to_datetime(ledger["EffectiveDate"])

    if snapshot_dates is None:
        # Use every date that appears in the ledger as a snapshot point
        snapshot_dates = sorted(ledger["EffectiveDate"].unique())

    rows = []
    for snap_date in snapshot_dates:
        snap_date = pd.Timestamp(snap_date)
        feats = _features_for_date(ledger, snap_date, config)
        if include_target:
            target = _build_target(ledger, snap_date, config)
            feats = feats.join(target, how="left")
            feats["called_next"] = feats["called_next"].fillna(0).astype(int)
        feats["snapshot_date"] = snap_date
        rows.append(feats)

    result = pd.concat(rows, axis=0)
    result = result.reset_index()  # brings AccountNumber back as a column
    result = result.sort_values(["AccountNumber", "snapshot_date"]).reset_index(drop=True)
    logger.info(
        "Feature matrix: %d rows, %d features, snapshot range %s – %s",
        len(result),
        result.shape[1],
        result["snapshot_date"].min().date(),
        result["snapshot_date"].max().date(),
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _features_for_date(
    ledger: pd.DataFrame,
    snap_date: pd.Timestamp,
    config: Config,
) -> pd.DataFrame:
    """
    Compute features for all accounts active as of snap_date.

    Returns a DataFrame indexed by AccountNumber.
    """
    max_lookback = max(config.lookback_windows)
    history_start = snap_date - pd.Timedelta(days=config.max_days_since)
    history_end = snap_date - pd.Timedelta(days=1)  # exclude snap_date itself

    hist = ledger[
        (ledger["EffectiveDate"] >= history_start)
        & (ledger["EffectiveDate"] <= history_end)
    ].copy()

    if hist.empty:
        return pd.DataFrame()

    active_accounts = hist["AccountNumber"].unique()

    feature_frames = []

    # ── Rolling window counts / amounts per category ──────────────────────
    for window in config.lookback_windows:
        window_start = snap_date - pd.Timedelta(days=window)
        w_hist = hist[hist["EffectiveDate"] >= window_start]

        for cat, label in [
            (config.category_inbound_call,  "inbound_calls"),
            (config.category_outbound_call, "outbound_calls"),
            (config.category_authorization, "authorizations"),
            (config.category_payment,       "payments"),
            (config.category_web_login,     "web_logins"),
        ]:
            cat_hist = w_hist[w_hist["Category"] == cat]

            counts = (
                cat_hist.groupby("AccountNumber")
                .size()
                .reindex(active_accounts, fill_value=0)
                .rename(f"{label}_{window}d")
            )
            feature_frames.append(counts)

            # Amount totals only make sense for financial categories
            if cat in (config.category_authorization, config.category_payment):
                amounts = (
                    cat_hist.groupby("AccountNumber")["Amount"]
                    .sum()
                    .reindex(active_accounts, fill_value=0.0)
                    .rename(f"{label}_amount_{window}d")
                )
                feature_frames.append(amounts)

    # ── Days-since-last-event per category ───────────────────────────────
    for cat, label in [
        (config.category_inbound_call,  "inbound_call"),
        (config.category_outbound_call, "outbound_call"),
        (config.category_authorization, "authorization"),
        (config.category_payment,       "payment"),
        (config.category_web_login,     "web_login"),
    ]:
        last_event = (
            hist[hist["Category"] == cat]
            .groupby("AccountNumber")["EffectiveDate"]
            .max()
        )
        days_since = (snap_date - last_event).dt.days.clip(upper=config.max_days_since)
        days_since = days_since.reindex(active_accounts, fill_value=config.max_days_since)
        days_since = days_since.rename(f"days_since_last_{label}")
        feature_frames.append(days_since)

    # ── Derived / ratio features ──────────────────────────────────────────
    feat_df = pd.concat(feature_frames, axis=1)

    # Call trend: 7-day rate vs 30-day rate (avoids div-by-zero)
    rate_7d  = feat_df["inbound_calls_7d"]  / 7.0
    rate_30d = feat_df["inbound_calls_30d"] / 30.0
    feat_df["call_rate_trend"] = (rate_7d - rate_30d).clip(-1.0, 1.0)

    # Self-service ratio: how often does the account use web vs call?
    total_30d = feat_df["inbound_calls_30d"] + feat_df["web_logins_30d"]
    feat_df["web_self_service_ratio_30d"] = np.where(
        total_30d > 0, feat_df["web_logins_30d"] / total_30d, 0.5
    )

    # Payment regularity: payments per 30-day window normalised to [0, 1]
    feat_df["payment_regularity"] = (feat_df["payments_30d"] / 4.0).clip(0, 1)

    # Total activity in past 30 days
    feat_df["total_events_30d"] = (
        feat_df["inbound_calls_30d"]
        + feat_df["outbound_calls_30d"]
        + feat_df["authorizations_30d"]
        + feat_df["payments_30d"]
        + feat_df["web_logins_30d"]
    )

    feat_df.index.name = "AccountNumber"
    return feat_df


def _build_target(
    ledger: pd.DataFrame,
    snap_date: pd.Timestamp,
    config: Config,
) -> pd.Series:
    """
    Binary target: 1 if the account placed an Inbound Call in the
    prediction horizon window after snap_date.
    """
    horizon_start = snap_date
    horizon_end   = snap_date + pd.Timedelta(days=config.prediction_horizon_days)

    future = ledger[
        (ledger["EffectiveDate"] > horizon_start)
        & (ledger["EffectiveDate"] <= horizon_end)
        & (ledger["Category"] == config.category_inbound_call)
    ]

    called = (
        future.groupby("AccountNumber")
        .size()
        .gt(0)
        .astype(int)
        .rename("called_next")
    )
    return called


# ---------------------------------------------------------------------------
# Convenience: feature column names (excludes meta-columns)
# ---------------------------------------------------------------------------

def feature_columns(df: pd.DataFrame) -> List[str]:
    """Return column names that are features (not meta or target)."""
    exclude = {"AccountNumber", "snapshot_date", "called_next"}
    return [c for c in df.columns if c not in exclude]
