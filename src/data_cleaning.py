"""Data loading and cleaning pipeline.

All data transformations are applied here. The original CSV is never modified.
Every transformation is logged so it can be explained and audited.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CSAT_MAX,
    CSAT_MIN,
    MARKET_NORMALIZATION,
    RAW_DATA_PATH,
)

logger = logging.getLogger(__name__)


def load_raw_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw ticket CSV and return an unmodified DataFrame.

    Args:
        path: Path to the CSV file. Defaults to RAW_DATA_PATH from config.

    Returns:
        Raw DataFrame as loaded from CSV.
    """
    path = Path(path) if path else RAW_DATA_PATH
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply all cleaning steps and return the cleaned DataFrame with a log.

    Cleaning steps (in order):
    1. Normalize market labels
    2. Clamp CSAT to valid range (1-5)
    3. Set negative resolution_min to NaN
    4. Parse timestamps
    5. Add derived columns

    Args:
        df: Raw DataFrame.

    Returns:
        Tuple of (clean DataFrame, cleaning log dict).
    """
    df = df.copy()
    cleaning_log: dict = {}

    # 1. Normalize market labels
    market_fixes = df["market"].isin(MARKET_NORMALIZATION.keys()).sum()
    df["market"] = df["market"].replace(MARKET_NORMALIZATION)
    cleaning_log["market_normalized"] = int(market_fixes)

    # 2. Clamp CSAT scores to valid range
    csat_out_of_range = (
        df["csat_score"].notna()
        & ((df["csat_score"] < CSAT_MIN) | (df["csat_score"] > CSAT_MAX))
    ).sum()
    df.loc[
        df["csat_score"].notna()
        & ((df["csat_score"] < CSAT_MIN) | (df["csat_score"] > CSAT_MAX)),
        "csat_score",
    ] = np.nan
    cleaning_log["csat_clamped"] = int(csat_out_of_range)

    # 3. Set negative resolution_min to NaN
    negative_resolution = (df["resolution_min"] < 0).sum()
    df.loc[df["resolution_min"] < 0, "resolution_min"] = np.nan
    cleaning_log["negative_resolution_fixed"] = int(negative_resolution)

    # 4. Parse timestamps
    df["created_at"] = pd.to_datetime(df["created_at"])

    # 5. Add derived columns
    df["week_number"] = df["created_at"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["created_at"].dt.day_name()
    df["hour_of_day"] = df["created_at"].dt.hour
    df["is_resolved"] = df["resolution_status"] == "resolved"
    df["is_business_hours"] = df["hour_of_day"].between(8, 18) & df[
        "day_of_week"
    ].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])

    logger.info("Cleaning complete. Log: %s", cleaning_log)
    return df, cleaning_log


def get_data_quality_report(
    raw_df: pd.DataFrame, clean_df: pd.DataFrame
) -> dict:
    """Generate a comprehensive data quality report.

    Args:
        raw_df: The original raw DataFrame.
        clean_df: The cleaned DataFrame.

    Returns:
        Dictionary with quality metrics.
    """
    total_cells = raw_df.shape[0] * raw_df.shape[1]
    missing_cells = raw_df.isnull().sum().sum() + (raw_df == "").sum().sum()

    missing_values = {}
    for col in raw_df.columns:
        null_count = raw_df[col].isnull().sum()
        empty_count = (raw_df[col] == "").sum() if raw_df[col].dtype == "object" else 0
        total_missing = null_count + empty_count
        if total_missing > 0:
            missing_values[col] = {
                "count": int(total_missing),
                "pct": round(total_missing / len(raw_df) * 100, 1),
            }

    return {
        "total_rows": len(raw_df),
        "total_columns": len(raw_df.columns),
        "date_range": {
            "start": str(clean_df["created_at"].min()),
            "end": str(clean_df["created_at"].max()),
        },
        "missing_values": missing_values,
        "completeness_score": round(
            (1 - missing_cells / total_cells) * 100, 1
        ),
    }
