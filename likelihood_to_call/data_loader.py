"""
Data loading utilities.

Reads the Ledger table from the configured SQL database and returns a
clean, typed pandas DataFrame ready for feature engineering.
"""
import logging
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

from .config import Config

logger = logging.getLogger(__name__)


def load_ledger(
    config: Config,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pull all rows from the Ledger table within the optional date range.

    Parameters
    ----------
    config : Config
    start_date : str, optional  e.g. "2023-01-01"
    end_date   : str, optional  e.g. "2024-12-31"

    Returns
    -------
    pd.DataFrame with columns:
        AccountNumber (str), EffectiveDate (date), Category (str), Amount (float)
    """
    engine = create_engine(config.db_connection_string)

    conditions = []
    params: dict = {}
    if start_date:
        conditions.append("EffectiveDate >= :start_date")
        params["start_date"] = start_date
    if end_date:
        conditions.append("EffectiveDate <= :end_date")
        params["end_date"] = end_date

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    query = text(
        f"SELECT AccountNumber, EffectiveDate, Category, Amount "
        f"FROM {config.ledger_table} {where_clause}"
    )

    logger.info("Loading ledger data from %s …", config.db_connection_string)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    df = _clean(df)
    logger.info("Loaded %d ledger rows spanning %s – %s", len(df), df["EffectiveDate"].min(), df["EffectiveDate"].max())
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce types and drop unusable rows."""
    df["EffectiveDate"] = pd.to_datetime(df["EffectiveDate"]).dt.normalize()
    df["AccountNumber"] = df["AccountNumber"].astype(str).str.strip()
    df["Category"] = df["Category"].str.strip()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["AccountNumber", "EffectiveDate", "Category"])
    return df.sort_values(["AccountNumber", "EffectiveDate"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic data generator – useful for unit tests and demos when no live DB
# is available.
# ---------------------------------------------------------------------------

def generate_synthetic_ledger(
    n_accounts: int = 500,
    start_date: str = "2023-01-01",
    end_date: str = "2024-06-30",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic Ledger DataFrame.

    High-activity accounts are seeded with more frequent events; a small
    cohort of low-activity accounts has an elevated probability of making
    surprise inbound calls (simulating operational events).

    Returns
    -------
    pd.DataFrame  (same schema as load_ledger)
    """
    import numpy as np

    rng = np.random.default_rng(random_state)
    dates = pd.date_range(start_date, end_date, freq="D")

    categories = [
        "Inbound Call",
        "Outbound Call",
        "Authorization",
        "Payment",
        "Web Login",
    ]

    # Base daily event rates per category per account type
    # Shape: [high_activity, low_activity]
    base_rates = {
        "Inbound Call":   [0.08, 0.02],
        "Outbound Call":  [0.05, 0.01],
        "Authorization":  [0.40, 0.10],
        "Payment":        [0.10, 0.04],
        "Web Login":      [0.25, 0.05],
    }

    # 70 % high-activity, 30 % low-activity
    account_ids = [f"ACC{i:05d}" for i in range(n_accounts)]
    account_type = rng.choice([0, 1], size=n_accounts, p=[0.70, 0.30])

    rows = []
    for acct_idx, (acct, atype) in enumerate(zip(account_ids, account_type)):
        for date in dates:
            for cat in categories:
                rate = base_rates[cat][atype]

                # Inject an "operational event" spike: on 5 specific dates,
                # low-activity accounts have a 10× higher inbound call rate.
                spike_dates = pd.to_datetime(
                    ["2023-06-15", "2023-09-01", "2023-11-20", "2024-01-10", "2024-04-05"]
                )
                if cat == "Inbound Call" and atype == 1 and date in spike_dates:
                    rate *= 10

                if rng.random() < rate:
                    amount = (
                        rng.uniform(5, 500) if cat in ("Authorization", "Payment") else 0.0
                    )
                    rows.append(
                        {
                            "AccountNumber": acct,
                            "EffectiveDate": date,
                            "Category": cat,
                            "Amount": round(float(amount), 2),
                        }
                    )

    df = pd.DataFrame(rows)
    df["EffectiveDate"] = pd.to_datetime(df["EffectiveDate"])
    return df.sort_values(["AccountNumber", "EffectiveDate"]).reset_index(drop=True)
