"""
Configuration for the Likelihood-to-Call scoring system.

All tuneable parameters live here so nothing is hardcoded in the modules.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Database                                                             #
    # ------------------------------------------------------------------ #
    # SQLAlchemy connection string.  Examples:
    #   SQLite  : "sqlite:///ledger.db"
    #   MSSQL   : "mssql+pyodbc://user:pw@server/db?driver=ODBC+Driver+17+for+SQL+Server"
    #   Postgres: "postgresql://user:pw@host:5432/db"
    db_connection_string: str = "sqlite:///ledger.db"
    ledger_table: str = "Ledger"

    # ------------------------------------------------------------------ #
    # Category labels in the Ledger table                                 #
    # ------------------------------------------------------------------ #
    category_inbound_call: str = "Inbound Call"
    category_outbound_call: str = "Outbound Call"
    category_authorization: str = "Authorization"
    category_payment: str = "Payment"
    category_web_login: str = "Web Login"

    # ------------------------------------------------------------------ #
    # Feature engineering                                                  #
    # ------------------------------------------------------------------ #
    # Rolling windows (in days) used to aggregate activity counts/amounts
    lookback_windows: List[int] = field(default_factory=lambda: [7, 30, 90])

    # Maximum days-since-last-event value (caps extreme values for accounts
    # that have been dormant a very long time)
    max_days_since: int = 180

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    # Binary target: did the account place an inbound call in the next N days?
    prediction_horizon_days: int = 7

    # Date used to split training vs hold-out evaluation.
    # None → use the latest 20 % of dates as the hold-out set.
    train_cutoff_date: Optional[str] = None

    # Path where the trained model artefact is persisted
    model_output_path: str = "likelihood_to_call_model.pkl"

    # ------------------------------------------------------------------ #
    # Spike detection                                                      #
    # ------------------------------------------------------------------ #
    # Accounts with predicted call-probability below this value are
    # considered "unlikely to call" (the cohort used for spike detection)
    low_prob_threshold: float = 0.20

    # Rolling window (days) used to estimate the baseline mean and std of
    # unexpected-caller counts
    spike_baseline_window: int = 30

    # Number of standard deviations above the rolling mean that triggers a
    # spike flag
    spike_z_threshold: float = 2.5

    # Minimum number of low-probability accounts that must be scored on a
    # given day before the spike signal is considered reliable
    min_low_prob_accounts: int = 30

    # ------------------------------------------------------------------ #
    # Misc                                                                 #
    # ------------------------------------------------------------------ #
    random_state: int = 42
