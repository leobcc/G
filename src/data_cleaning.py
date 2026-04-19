"""Data loading and cleaning pipeline.

All data transformations are applied here. The original CSV is never modified.
Every transformation is logged so it can be explained and audited.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder

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
    df = pd.read_csv(path, low_memory=False)
    # Drop rows where ALL values are NaN (common with Excel-exported CSVs
    # that pad to 1,048,576 rows)
    df = df.dropna(how="all").reset_index(drop=True)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def _impute_median_fallback(
    df: pd.DataFrame, targets: list[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fast median imputation fallback for large datasets or missing context.

    Groups by available categorical dimensions (category, assigned_team)
    and fills missing target values with group medians. Falls back to global
    median where groups are too small.
    """
    imputation_log: dict[str, int] = {}
    group_cols = [c for c in ["category", "assigned_team"] if c in df.columns]

    for col in targets:
        was_missing = df[col].isnull()
        imputed_count = int(was_missing.sum())
        if imputed_count == 0:
            df[f"{col}_imputed"] = False
            imputation_log[f"{col}_imputed"] = 0
            continue

        if group_cols:
            group_medians = df.groupby(group_cols)[col].transform("median")
            df[col] = df[col].fillna(group_medians)

        # Fill remaining NaN with global median
        global_median = df[col].median()
        if pd.notna(global_median):
            df[col] = df[col].fillna(global_median)

        # Post-process
        if col == "csat_score":
            df[col] = df[col].clip(CSAT_MIN, CSAT_MAX).round(0)
        else:
            df[col] = df[col].clip(lower=0)

        df[f"{col}_imputed"] = was_missing & df[col].notna()
        final_imputed = int(df[f"{col}_imputed"].sum())
        imputation_log[f"{col}_imputed"] = final_imputed
        logger.info("Imputed %d missing %s values via median fallback", final_imputed, col)

    # Restore structural NaN for abandoned/pending
    if "resolution_status" in df.columns and "resolution_min" in targets:
        structurally_absent = df["resolution_status"].isin(["abandoned", "pending"])
        df.loc[structurally_absent, "resolution_min"] = np.nan
        df.loc[structurally_absent, "resolution_min_imputed"] = False

    return df, imputation_log


def _impute_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Impute missing csat_score, resolution_min, first_response_min using KNN.

    Uses K-Nearest Neighbors imputation (k=5) based on the surrounding
    feature context of each ticket: channel, category, priority, team,
    market, resolution_status, contacts_per_ticket, cost_usd, and
    time-based features.

    The approach:
    1. Ordinal-encode categorical features so KNN can compute distances.
    2. Build a feature matrix of encoded categoricals + available numerics.
    3. Run KNNImputer (k=5, distance-weighted) to fill gaps.
    4. Post-process: clamp CSAT to [1, 5], clamp times to >= 0, round CSAT
       to nearest integer.
    5. Restore resolution_min to NaN for abandoned/pending tickets
       (structurally absent — these tickets were never resolved).
    6. Add boolean flags (``*_imputed``) so downstream code can distinguish
       original from imputed values.

    Args:
        df: DataFrame after basic cleaning (markets normalized, timestamps
            parsed, derived columns added).  Modified **in-place** for
            efficiency, but a copy is returned for clarity.

    Returns:
        Tuple of (DataFrame with imputed values + imputation flags,
        dict mapping column name → count of values imputed).
    """
    targets = ["csat_score", "resolution_min", "first_response_min"]
    # Only target columns that actually exist
    targets = [t for t in targets if t in df.columns]
    if not targets:
        return df, {}

    missing_before = {col: int(df[col].isnull().sum()) for col in targets}

    # If nothing to impute, short-circuit
    if all(v == 0 for v in missing_before.values()):
        for col in targets:
            df[f"{col}_imputed"] = False
        return df, {f"{col}_imputed": 0 for col in targets}

    # For very large datasets (>50K rows), use median imputation instead of KNN
    # to avoid excessive memory/time. KNN is O(n²) in practice.
    if len(df) > 50_000:
        logger.info(
            "Dataset has %d rows — using median imputation instead of KNN for performance",
            len(df),
        )
        return _impute_median_fallback(df, targets)

    # ── Feature matrix for KNN ────────────────────────────────────────────
    cat_features = ["channel", "category", "priority", "assigned_team",
                    "market", "resolution_status"]
    num_features = ["contacts_per_ticket", "cost_usd", "hour_of_day"]

    # Only use cat features that exist in the DataFrame
    available_cat = [f for f in cat_features if f in df.columns]
    # Add any available numeric features that aren't targets
    available_num = [f for f in num_features if f in df.columns]

    # Need at least some context features for KNN to work
    if not available_cat and not available_num:
        logger.warning("No context features available for KNN — using median imputation")
        return _impute_median_fallback(df, targets)

    # Ordinal-encode categoricals
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                             unknown_value=-1)
    cat_encoded = encoder.fit_transform(df[available_cat].fillna("missing")) if available_cat else np.empty((len(df), 0))

    # Build the combined feature + target matrix
    feature_cols = [f"_enc_{c}" for c in available_cat] + available_num + targets
    impute_df = pd.DataFrame(
        cat_encoded,
        columns=[f"_enc_{c}" for c in available_cat],
        index=df.index,
    )
    for col in available_num:
        impute_df[col] = df[col].values
    for col in targets:
        impute_df[col] = df[col].values

    # ── KNN Imputation ────────────────────────────────────────────────────
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    imputed_array = imputer.fit_transform(impute_df[feature_cols])

    # Extract only the target columns (last 3 columns in feature_cols)
    target_indices = [feature_cols.index(t) for t in targets]

    imputation_log: dict[str, int] = {}
    for col, idx in zip(targets, target_indices):
        was_missing = df[col].isnull()
        imputed_values = imputed_array[:, idx]

        # Post-process imputed values
        if col == "csat_score":
            # CSAT scores are integers 1-5; round to nearest int
            imputed_values = np.clip(np.round(imputed_values).astype(int), CSAT_MIN, CSAT_MAX)
        else:
            # Times must be non-negative, round to 1 decimal
            imputed_values = np.clip(np.round(imputed_values, 1), 0, None)

        # Apply only to originally missing values
        df.loc[was_missing, col] = imputed_values[was_missing.values]
        df[f"{col}_imputed"] = was_missing

        imputed_count = int(was_missing.sum())
        imputation_log[f"{col}_imputed"] = imputed_count
        logger.info("Imputed %d missing %s values via KNN (k=5)", imputed_count, col)

    # Correct resolution_min for abandoned/pending tickets — these are
    # structurally absent (never resolved), not missing data.
    if "resolution_status" in df.columns:
        structurally_absent = df["resolution_status"].isin(["abandoned", "pending"])
        n_restored = int((structurally_absent & df["resolution_min_imputed"]).sum())
        df.loc[structurally_absent, "resolution_min"] = np.nan
        df.loc[structurally_absent, "resolution_min_imputed"] = False
        imputation_log["resolution_min_imputed"] -= n_restored
        logger.info(
            "Restored %d abandoned/pending resolution_min to NaN (structurally absent)",
            n_restored,
        )

    return df, imputation_log


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply all cleaning steps and return the cleaned DataFrame with a log.

    Cleaning steps (in order):
    1. Normalize market labels
    2. Clamp CSAT to valid range (1-5)
    3. Set negative resolution_min to NaN
    4. Fill structural placeholders (agent_id for chatbot, subcategory)
    5. Parse timestamps
    6. Add derived columns
    7. KNN imputation for CSAT, FRT, resolution_min

    Args:
        df: Raw DataFrame.

    Returns:
        Tuple of (clean DataFrame, cleaning log dict).
    """
    df = df.copy()
    cleaning_log: dict = {}

    # 1. Normalize market labels (guard: only if 'market' column exists)
    if "market" in df.columns:
        market_fixes = df["market"].isin(MARKET_NORMALIZATION.keys()).sum()
        df["market"] = df["market"].replace(MARKET_NORMALIZATION)
        cleaning_log["market_normalized"] = int(market_fixes)

    # 2. Clamp CSAT scores to valid range (guard: only if column exists)
    if "csat_score" in df.columns:
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

    # 3. Set negative resolution_min to NaN (guard: only if column exists)
    if "resolution_min" in df.columns:
        neg_mask = df["resolution_min"].notna() & (df["resolution_min"] < 0)
        negative_resolution = neg_mask.sum()
        df.loc[neg_mask, "resolution_min"] = np.nan
        cleaning_log["negative_resolution_fixed"] = int(negative_resolution)

    # 4. Fill structural placeholders (before imputation)
    #    - agent_id is structurally absent for chatbot tickets
    #    - subcategory has organic NaNs that should be "unspecified"
    if "agent_id" in df.columns and "assigned_team" in df.columns:
        chatbot_mask = df["assigned_team"] == "ai_chatbot"
        agent_id_filled = int(chatbot_mask.sum() - df.loc[chatbot_mask, "agent_id"].notna().sum())
        df.loc[chatbot_mask & df["agent_id"].isna(), "agent_id"] = "chatbot_bot"
        cleaning_log["agent_id_placeholder"] = agent_id_filled
    if "subcategory" in df.columns:
        subcat_filled = int(df["subcategory"].isna().sum())
        df["subcategory"] = df["subcategory"].fillna("unspecified")
        cleaning_log["subcategory_placeholder"] = subcat_filled

    # 5. Parse timestamps (coerce unparseable values → NaT)
    #    Try multiple strategies to handle different date formats:
    #    1. Default pd.to_datetime inference
    #    2. dayfirst=True for DD/MM/YYYY formats
    #    3. Common explicit formats
    if "created_at" in df.columns:
        parsed = pd.to_datetime(df["created_at"], errors="coerce")
        # If >50% failed, try dayfirst=True (common in UK/EU data)
        nat_ratio = parsed.isna().sum() / max(len(parsed), 1)
        if nat_ratio > 0.5:
            parsed_alt = pd.to_datetime(df["created_at"], dayfirst=True, errors="coerce")
            if parsed_alt.isna().sum() < parsed.isna().sum():
                parsed = parsed_alt
        # If still >50% failed, try common explicit formats
        if parsed.isna().sum() / max(len(parsed), 1) > 0.5:
            for fmt in ("%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
                        "%Y/%m/%d %H:%M:%S", "%d-%m-%Y %H:%M:%S",
                        "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M",
                        "%d/%m/%Y", "%m/%d/%Y"):
                attempt = pd.to_datetime(df["created_at"], format=fmt, errors="coerce")
                if attempt.isna().sum() < parsed.isna().sum():
                    parsed = attempt
                    break
        df["created_at"] = parsed
        nat_count = int(df["created_at"].isna().sum())
        if nat_count > 0:
            cleaning_log["unparseable_timestamps"] = nat_count
            logger.warning("%d rows have unparseable timestamps (set to NaT)", nat_count)

    # 6. Add derived columns (safe with NaT — produces NA which is handled downstream)
    # Use nullable Int64 so NA values don't crash .astype(int)
    iso_week = df["created_at"].dt.isocalendar().week
    df["week_number"] = iso_week.astype("Int64")
    df["day_of_week"] = df["created_at"].dt.day_name()
    df["hour_of_day"] = df["created_at"].dt.hour
    df["is_resolved"] = df["resolution_status"] == "resolved" if "resolution_status" in df.columns else False
    # is_business_hours: safe even with NaN hour/day — .between() and .isin() return False for NaN
    df["is_business_hours"] = (
        df["hour_of_day"].between(8, 18)
        & df["day_of_week"].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    )

    # 7. Impute missing csat_score, resolution_min, first_response_min via KNN
    df, imputation_log = _impute_missing_values(df)
    cleaning_log.update(imputation_log)

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
            1 - missing_cells / total_cells, 4
        ),
    }


def detect_complete_weeks(df: pd.DataFrame, threshold: float = 0.35) -> list[int]:
    """Identify complete (non-partial) weeks in the dataset.

    A week is considered "partial" if it has significantly fewer tickets
    than the median week — typically the first or last week of the data range.

    Args:
        df: Clean DataFrame with ``week_number`` column.
        threshold: Fraction of median volume below which a week is
            considered partial.  Default 0.35 (35%).

    Returns:
        Sorted list of complete week numbers.  Falls back to all available
        weeks if the calculation yields fewer than 2 weeks.
    """
    if "week_number" not in df.columns:
        logger.warning("No week_number column — returning empty week list")
        return []

    # Determine date column for span calculation
    date_col = None
    for col in ("created_at", "timestamp", "date"):
        if col in df.columns:
            date_col = col
            break

    if date_col is None or not pd.api.types.is_datetime64_any_dtype(df.get(date_col, pd.Series())):
        # Fallback to volume-based if no usable datetime column
        counts = df.groupby("week_number").size()
        if counts.empty:
            return []
        all_weeks = sorted(counts.index.dropna().astype(int).tolist())
        if len(all_weeks) <= 1:
            return all_weeks
        median_count = counts.median()
        cutoff = median_count * threshold
        complete = sorted([
            int(wk) for wk, cnt in counts.items()
            if pd.notna(wk) and cnt >= cutoff
        ])
        if len(complete) < 2 and len(all_weeks) >= 2:
            # Take the top 2 by volume, not ALL weeks
            ranked = counts.sort_values(ascending=False)
            complete = sorted(int(wk) for wk in ranked.index[:2])
        logger.info(
            "Detected %d complete weeks (volume-based fallback): %s",
            len(complete), complete,
        )
        return complete

    # Date-span-based detection: a complete week must have data spanning
    # a significant portion of the days in a typical full week.
    # We use the max day-count as reference and keep weeks with ≥40% of that
    # (i.e. ≥3 days when the best week has 7).  This avoids both the
    # first and last partial weeks automatically.

    work = df[["week_number", date_col]].dropna(subset=["week_number", date_col]).copy()
    work["_date"] = pd.to_datetime(work[date_col]).dt.date

    # Count distinct days per week
    day_spans = work.groupby("week_number")["_date"].nunique()

    all_weeks = sorted(day_spans.index.dropna().astype(int).tolist())
    if len(all_weeks) <= 1:
        return all_weeks

    max_days = day_spans.max()
    # Dynamic threshold: at least 40% of the fullest week (min 3 days absolute)
    day_cutoff = max(3, int(max_days * 0.4))

    complete = sorted([
        int(wk)
        for wk, n_days in day_spans.items()
        if pd.notna(wk) and n_days >= day_cutoff
    ])

    # If fewer than 2 weeks pass, take the top 2 by day count instead of
    # blindly including ALL weeks (which would pull in truly partial ones).
    if len(complete) < 2 and len(all_weeks) >= 2:
        ranked = day_spans.sort_values(ascending=False)
        complete = sorted(int(wk) for wk in ranked.index[:2])

    logger.info(
        "Detected %d complete weeks out of %d total (date-span, cutoff %d days): %s",
        len(complete), len(all_weeks), day_cutoff, complete,
    )
    return complete
