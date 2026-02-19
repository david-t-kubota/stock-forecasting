"""
End-to-end pipeline: train the likelihood-to-call model, score accounts,
detect operational spikes, and export results.

Usage
-----
# Train on data from your SQL database and write outputs to CSV:
    python pipeline.py --mode train --start-date 2023-01-01 --end-date 2024-06-30

# Score today's accounts using a previously trained model:
    python pipeline.py --mode score --model-path likelihood_to_call_model.pkl

# Use synthetic data for a full demo / smoke-test:
    python pipeline.py --mode demo

# Re-detect spikes from a saved scores CSV (no retraining):
    python pipeline.py --mode spikes --scores-path scores.csv --ledger-path ledger.csv
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from likelihood_to_call.config import Config
from likelihood_to_call.data_loader import generate_synthetic_ledger, load_ledger
from likelihood_to_call.features import build_feature_matrix, feature_columns
from likelihood_to_call.model import (
    feature_importance,
    load_model,
    save_model,
    score,
    train,
)
from likelihood_to_call.spike_detector import detect_spikes, plot_spike_chart, spike_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_demo(config: Config, output_dir: Path) -> None:
    """Full smoke-test using synthetic data – no database required."""
    logger.info("=== DEMO MODE (synthetic data) ===")

    # 1. Generate data
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating synthetic ledger …")
    ledger = generate_synthetic_ledger(
        n_accounts=300,
        start_date="2023-01-01",
        end_date="2024-06-30",
        random_state=config.random_state,
    )
    ledger_path = output_dir / "ledger_synthetic.csv"
    ledger.to_csv(ledger_path, index=False)
    logger.info("Synthetic ledger saved → %s", ledger_path)

    _train_score_detect(ledger, config, output_dir)


def run_train(
    config: Config,
    output_dir: Path,
    start_date: str = None,
    end_date: str = None,
) -> None:
    """Train on data from the configured SQL database."""
    logger.info("=== TRAIN MODE ===")
    ledger = load_ledger(config, start_date=start_date, end_date=end_date)
    _train_score_detect(ledger, config, output_dir)


def _train_score_detect(
    ledger: pd.DataFrame,
    config: Config,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Build feature matrix (for all snapshot dates)
    logger.info("Building feature matrix …")
    snapshot_dates = _select_snapshot_dates(ledger, config)
    feature_matrix = build_feature_matrix(
        ledger, config, snapshot_dates=snapshot_dates, include_target=True
    )
    feature_matrix.to_csv(output_dir / "feature_matrix.csv", index=False)

    # 3. Train model
    logger.info("Training model …")
    model, metrics = train(feature_matrix, config)

    logger.info("Evaluation metrics:")
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)

    model_path = str(output_dir / "likelihood_to_call_model.pkl")
    save_model(model, model_path)

    # Feature importance
    imp = feature_importance(model, feature_matrix)
    if not imp.empty:
        imp.to_csv(output_dir / "feature_importance.csv", index=False)
        logger.info("Top 10 features:\n%s", imp.head(10).to_string(index=False))

    # 4. Score all snapshot dates
    logger.info("Scoring all accounts …")
    scores = score(model, feature_matrix)
    scores_path = output_dir / "scores.csv"
    scores.to_csv(scores_path, index=False)
    logger.info("Scores saved → %s", scores_path)

    # 5. Detect spikes
    logger.info("Detecting operational spikes …")
    daily = detect_spikes(scores, ledger, config)
    daily_path = output_dir / "spike_detection.csv"
    daily.to_csv(daily_path, index=False)
    logger.info("Spike detection saved → %s", daily_path)

    summary = spike_summary(daily)
    if not summary.empty:
        spike_path = output_dir / "spike_summary.csv"
        summary.to_csv(spike_path, index=False)
        logger.info("Spike summary (%d days flagged):\n%s", len(summary), summary.to_string(index=False))
    else:
        logger.info("No spikes detected – no spike_summary.csv written.")

    # 6. Plot
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        plot_spike_chart(daily, metric="actual_unexpected_calls", ax=axes[0])
        plot_spike_chart(daily, metric="unexpected_rate", ax=axes[1])
        plt.tight_layout()
        chart_path = output_dir / "spike_chart.png"
        plt.savefig(chart_path, dpi=120)
        plt.close()
        logger.info("Chart saved → %s", chart_path)
    except Exception as exc:
        logger.warning("Could not generate chart: %s", exc)


def run_score(
    config: Config,
    output_dir: Path,
    model_path: str,
    snapshot_date: str = None,
    start_date: str = None,
    end_date: str = None,
) -> None:
    """Score accounts using an existing model."""
    logger.info("=== SCORE MODE ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path)
    ledger = load_ledger(config, start_date=start_date, end_date=end_date)

    if snapshot_date:
        snap_dates = [pd.Timestamp(snapshot_date)]
    else:
        snap_dates = _select_snapshot_dates(ledger, config)

    feature_matrix = build_feature_matrix(
        ledger, config, snapshot_dates=snap_dates, include_target=False
    )

    scores = score(model, feature_matrix)
    scores_path = output_dir / "scores.csv"
    scores.to_csv(scores_path, index=False)
    logger.info("Scores saved → %s", scores_path)


def run_spikes(
    config: Config,
    output_dir: Path,
    scores_path: str,
    ledger_path: str,
) -> None:
    """Re-run spike detection on previously saved scores and ledger CSVs."""
    logger.info("=== SPIKES MODE ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(scores_path, parse_dates=["snapshot_date"])
    ledger = pd.read_csv(ledger_path, parse_dates=["EffectiveDate"])

    daily = detect_spikes(scores, ledger, config)
    daily.to_csv(output_dir / "spike_detection.csv", index=False)

    summary = spike_summary(daily)
    if not summary.empty:
        summary.to_csv(output_dir / "spike_summary.csv", index=False)
        print(summary.to_string(index=False))
    else:
        logger.info("No spikes found.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_snapshot_dates(ledger: pd.DataFrame, config: Config) -> list:
    """
    Choose the snapshot dates to build features on.

    We skip the first `max(lookback_windows)` days (no history yet) and
    the last `prediction_horizon_days` days (target unknown at training time).
    """
    all_dates = sorted(ledger["EffectiveDate"].dt.normalize().unique())
    warmup = max(config.lookback_windows)
    if len(all_dates) <= warmup:
        return all_dates
    # Skip warmup at the start; keep all dates (target leakage handled in features.py)
    return all_dates[warmup:]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Likelihood-to-Call scoring pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["demo", "train", "score", "spikes"],
        default="demo",
        help="Pipeline mode",
    )
    p.add_argument("--output-dir", default="outputs", help="Directory for all output files")
    p.add_argument("--db-url", default=None, help="SQLAlchemy connection string (overrides config)")
    p.add_argument("--start-date", default=None, help="Ledger pull start date (YYYY-MM-DD)")
    p.add_argument("--end-date",   default=None, help="Ledger pull end date   (YYYY-MM-DD)")
    p.add_argument("--snapshot-date", default=None, help="Single scoring date (score mode)")
    p.add_argument("--model-path", default="outputs/likelihood_to_call_model.pkl",
                   help="Path to saved model (score mode)")
    p.add_argument("--scores-path", default="outputs/scores.csv", help="Scores CSV (spikes mode)")
    p.add_argument("--ledger-path", default="outputs/ledger_synthetic.csv",
                   help="Ledger CSV (spikes mode)")
    p.add_argument("--low-prob-threshold", type=float, default=0.20,
                   help="Call-probability cutoff for low-probability cohort")
    p.add_argument("--spike-z", type=float, default=2.5, help="Z-score spike threshold")
    p.add_argument("--horizon-days", type=int, default=7, help="Prediction horizon in days")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    config = Config(
        low_prob_threshold=args.low_prob_threshold,
        spike_z_threshold=args.spike_z,
        prediction_horizon_days=args.horizon_days,
    )
    if args.db_url:
        config.db_connection_string = args.db_url

    output_dir = Path(args.output_dir)

    if args.mode == "demo":
        run_demo(config, output_dir)
    elif args.mode == "train":
        run_train(config, output_dir, start_date=args.start_date, end_date=args.end_date)
    elif args.mode == "score":
        run_score(
            config, output_dir,
            model_path=args.model_path,
            snapshot_date=args.snapshot_date,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    elif args.mode == "spikes":
        run_spikes(config, output_dir, args.scores_path, args.ledger_path)
    else:
        logger.error("Unknown mode: %s", args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
