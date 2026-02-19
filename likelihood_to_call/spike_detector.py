"""
Operational spike detection.

Goal
----
Identify calendar days on which an unusually high number of accounts that
the model scored as "unlikely to call" nevertheless placed inbound calls.
Such spikes are a signal that something happened operationally (e.g. a
billing error, statement rendering bug, payment posting failure, or
collections contact storm) that caused customers who normally handle things
themselves — or who are simply not in a distressed state — to reach out.

Algorithm
---------
For each date d in the scoring history:

1. Take the set of accounts scored on d with call_probability < threshold.
2. Count how many of those accounts actually placed an inbound call on d
   (actual_unexpected_calls).
3. Compute a second normalised metric:
       unexpected_rate = actual_unexpected_calls / n_low_prob_accounts
4. Fit a rolling baseline (mean ± std) over the preceding `baseline_window`
   days for both metrics.
5. Flag d as a spike if either metric's z-score > spike_z_threshold, provided
   at least `min_low_prob_accounts` accounts were scored.

Output
------
A DataFrame with one row per date containing the raw counts, the rolling
statistics, z-scores, and a boolean `is_spike` flag.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_spikes(
    scores: pd.DataFrame,
    ledger: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """
    Detect days with an anomalous number of calls from low-probability accounts.

    Parameters
    ----------
    scores : pd.DataFrame
        Output of model.score() – must have columns:
            AccountNumber, snapshot_date, call_probability
    ledger : pd.DataFrame
        Raw ledger data (same date range as scores).
    config : Config

    Returns
    -------
    pd.DataFrame  (one row per date in scores, sorted chronologically)
        Columns:
            date
            n_low_prob_accounts      – accounts scored with prob < threshold
            actual_unexpected_calls  – how many of those actually called
            unexpected_rate          – actual_unexpected_calls / n_low_prob_accounts
            expected_calls           – sum of probabilities (baseline expectation)
            observed_vs_expected     – actual / expected ratio
            unexpected_calls_mean    – rolling mean of actual_unexpected_calls
            unexpected_calls_std     – rolling std
            unexpected_rate_mean     – rolling mean of unexpected_rate
            unexpected_rate_std      – rolling std
            unexpected_calls_zscore  – z-score of actual_unexpected_calls
            unexpected_rate_zscore   – z-score of unexpected_rate
            is_spike                 – True if either z-score > threshold
            reliable                 – True if n_low_prob_accounts >= minimum
    """
    scores = scores.copy()
    scores["snapshot_date"] = pd.to_datetime(scores["snapshot_date"])

    ledger = ledger.copy()
    ledger["EffectiveDate"] = pd.to_datetime(ledger["EffectiveDate"])

    # Inbound calls only
    calls = ledger[ledger["Category"] == config.category_inbound_call][
        ["AccountNumber", "EffectiveDate"]
    ].drop_duplicates()

    daily_rows = []

    for date, day_scores in scores.groupby("snapshot_date"):
        low_prob = day_scores[
            day_scores["call_probability"] < config.low_prob_threshold
        ]
        n_low = len(low_prob)

        # Accounts in the low-prob cohort that actually called on this date
        actual_callers = calls[calls["EffectiveDate"] == date]["AccountNumber"]
        unexpected_count = low_prob["AccountNumber"].isin(actual_callers).sum()
        unexpected_rate  = unexpected_count / n_low if n_low > 0 else np.nan

        # Expected call count = sum of individual probabilities (Poisson approx)
        expected = low_prob["call_probability"].sum()
        obs_vs_exp = unexpected_count / expected if expected > 0 else np.nan

        daily_rows.append(
            {
                "date":                     pd.Timestamp(date),
                "n_low_prob_accounts":      n_low,
                "actual_unexpected_calls":  int(unexpected_count),
                "unexpected_rate":          unexpected_rate,
                "expected_calls":           round(float(expected), 2),
                "observed_vs_expected":     round(float(obs_vs_exp), 4) if not np.isnan(obs_vs_exp) else np.nan,
            }
        )

    daily = (
        pd.DataFrame(daily_rows)
        .sort_values("date")
        .reset_index(drop=True)
    )

    if daily.empty:
        return daily

    # ── Rolling baseline and z-scores ────────────────────────────────────
    w = config.spike_baseline_window

    for metric in ("actual_unexpected_calls", "unexpected_rate"):
        roll = daily[metric].rolling(window=w, min_periods=max(1, w // 2))
        daily[f"{metric}_mean"] = roll.mean()
        daily[f"{metric}_std"]  = roll.std().fillna(0)

        std_safe = daily[f"{metric}_std"].replace(0, np.nan)
        daily[f"{metric}_zscore"] = (
            (daily[metric] - daily[f"{metric}_mean"]) / std_safe
        ).fillna(0)

    # ── Spike flag ────────────────────────────────────────────────────────
    daily["reliable"] = daily["n_low_prob_accounts"] >= config.min_low_prob_accounts

    daily["is_spike"] = (
        daily["reliable"]
        & (
            (daily["actual_unexpected_calls_zscore"] > config.spike_z_threshold)
            | (daily["unexpected_rate_zscore"]         > config.spike_z_threshold)
        )
    )

    spike_dates = daily.loc[daily["is_spike"], "date"].dt.date.tolist()
    if spike_dates:
        logger.info("Spike dates detected (%d): %s", len(spike_dates), spike_dates)
    else:
        logger.info("No operational spikes detected.")

    return daily


def spike_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the flagged spike days with their key statistics.
    """
    cols = [
        "date",
        "n_low_prob_accounts",
        "actual_unexpected_calls",
        "unexpected_rate",
        "expected_calls",
        "observed_vs_expected",
        "actual_unexpected_calls_zscore",
        "unexpected_rate_zscore",
    ]
    available = [c for c in cols if c in daily.columns]
    return daily.loc[daily["is_spike"], available].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_spike_chart(
    daily: pd.DataFrame,
    metric: str = "actual_unexpected_calls",
    title: Optional[str] = None,
    ax=None,
):
    """
    Plot the daily metric with its rolling mean ± 2.5 std band and spike flags.

    Parameters
    ----------
    daily  : output of detect_spikes()
    metric : "actual_unexpected_calls" or "unexpected_rate"
    title  : optional axis title
    ax     : matplotlib Axes (created if None)

    Returns
    -------
    matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        raise ImportError("matplotlib is required for plotting.")

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))

    mean_col = f"{metric}_mean"
    std_col  = f"{metric}_std"
    z_col    = f"{metric}_zscore"

    ax.plot(daily["date"], daily[metric], color="steelblue", lw=1.2, label=metric)
    ax.plot(daily["date"], daily[mean_col], color="grey", lw=1, ls="--", label="Rolling mean")

    upper = daily[mean_col] + 2.5 * daily[std_col]
    lower = (daily[mean_col] - 2.5 * daily[std_col]).clip(lower=0)
    ax.fill_between(daily["date"], lower, upper, alpha=0.15, color="grey", label="±2.5 σ band")

    spike_mask = daily["is_spike"] & (daily[f"actual_unexpected_calls_zscore"] > 0)
    spikes = daily[daily["is_spike"]]
    ax.scatter(
        spikes["date"], spikes[metric],
        color="crimson", zorder=5, s=60, label="Spike flagged"
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    ax.set_xlabel("Date")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Operational Spike Detection – {metric}")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    return ax
