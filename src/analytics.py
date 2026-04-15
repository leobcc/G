"""Statistical analysis functions for the Ops Intelligence system.

Each function takes a clean DataFrame and returns structured results.
All functions are pure — no side effects, no LLM calls.
"""

import logging

import numpy as np
import pandas as pd

from src.config import COMPLETE_WEEKS

logger = logging.getLogger(__name__)


def compute_kpi_summary(df: pd.DataFrame, week: int | None = None) -> dict:
    """Compute overall KPI summary for the dataset or a specific week.

    Args:
        df: Clean DataFrame.
        week: Optional week number to filter by.

    Returns:
        Dictionary of KPI values.
    """
    if week is not None:
        df = df[df["week_number"] == week]

    total = len(df)
    if total == 0:
        return {}

    resolved = df["is_resolved"].sum()
    escalated = (df["resolution_status"] == "escalated").sum()
    abandoned = (df["resolution_status"] == "abandoned").sum()

    frt = df["first_response_min"].dropna()
    res = df.loc[df["resolution_min"].notna() & (df["resolution_min"] >= 0), "resolution_min"]
    csat = df.loc[df["csat_score"].between(1, 5), "csat_score"]

    return {
        "total_tickets": int(total),
        "avg_first_response_min": round(frt.mean(), 1) if len(frt) else None,
        "median_first_response_min": round(frt.median(), 1) if len(frt) else None,
        "avg_resolution_min": round(res.mean(), 1) if len(res) else None,
        "median_resolution_min": round(res.median(), 1) if len(res) else None,
        "resolution_rate": round(resolved / total, 3),
        "escalation_rate": round(escalated / total, 3),
        "abandonment_rate": round(abandoned / total, 3),
        "avg_csat": round(csat.mean(), 2) if len(csat) else None,
        "csat_collection_rate": round(len(csat) / total, 3),
        "avg_cost_usd": round(df["cost_usd"].mean(), 2),
        "total_cost_usd": round(df["cost_usd"].sum(), 2),
        "avg_contacts_per_ticket": round(df["contacts_per_ticket"].mean(), 1),
    }


def compute_metric_by_dimension(
    df: pd.DataFrame,
    metric: str,
    dimension: str,
    agg: str = "mean",
) -> pd.DataFrame:
    """Compute a metric grouped by a dimension.

    Args:
        df: Clean DataFrame.
        metric: Column name of the metric to aggregate.
        dimension: Column name to group by.
        agg: Aggregation function ('mean', 'median', 'sum', 'count').

    Returns:
        DataFrame with dimension values and aggregated metric, sorted descending.
    """
    result = df.groupby(dimension)[metric].agg(agg).reset_index()
    result.columns = [dimension, f"{metric}_{agg}"]
    return result.sort_values(f"{metric}_{agg}", ascending=False)


def compare_weeks(
    df: pd.DataFrame, metric: str, week_a: int, week_b: int
) -> dict:
    """Compare a metric between two weeks.

    Args:
        df: Clean DataFrame.
        metric: Column name or computed metric.
        week_a: Current week number.
        week_b: Prior week number.

    Returns:
        Dict with comparison results.
    """
    df_a = df[df["week_number"] == week_a]
    df_b = df[df["week_number"] == week_b]

    val_a = df_a[metric].mean() if metric in df_a.columns else 0
    val_b = df_b[metric].mean() if metric in df_b.columns else 0

    if val_b != 0:
        pct_change = round((val_a - val_b) / abs(val_b) * 100, 1)
    else:
        pct_change = 0.0

    abs_change = round(val_a - val_b, 2)

    # Determine direction (context-dependent — lower FRT is better, higher CSAT is better)
    improving_when_lower = {"first_response_min", "resolution_min", "cost_usd", "contacts_per_ticket"}
    if metric in improving_when_lower:
        direction = "improving" if abs_change < 0 else ("worsening" if abs_change > 0 else "stable")
    else:
        direction = "improving" if abs_change > 0 else ("worsening" if abs_change < 0 else "stable")

    return {
        "week_a_value": round(val_a, 2),
        "week_b_value": round(val_b, 2),
        "absolute_change": abs_change,
        "pct_change": pct_change,
        "direction": direction,
    }


def compute_weekly_trends(
    df: pd.DataFrame, metrics: list[str]
) -> pd.DataFrame:
    """Compute metrics for each complete week.

    Args:
        df: Clean DataFrame.
        metrics: List of column names to aggregate.

    Returns:
        DataFrame with weeks as rows and metrics as columns.
    """
    weekly = df[df["week_number"].isin(COMPLETE_WEEKS)]
    result = weekly.groupby("week_number")[metrics].mean().reset_index()
    return result


def compute_team_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive team performance comparison.

    Returns DataFrame with one row per team and columns for each metric.
    """
    teams = df.groupby("assigned_team")
    records = []

    for team_name, team_df in teams:
        total = len(team_df)
        resolved = team_df["is_resolved"].sum()
        frt = team_df["first_response_min"].dropna()
        res_time = team_df.loc[team_df["resolution_min"].notna() & (team_df["resolution_min"] >= 0), "resolution_min"]
        csat = team_df.loc[team_df["csat_score"].between(1, 5), "csat_score"]

        cost_resolved = team_df.loc[team_df["is_resolved"], "cost_usd"].sum()

        records.append({
            "team": team_name,
            "ticket_count": total,
            "resolution_rate": round(resolved / total, 3) if total else 0,
            "escalation_rate": round((team_df["resolution_status"] == "escalated").sum() / total, 3) if total else 0,
            "abandonment_rate": round((team_df["resolution_status"] == "abandoned").sum() / total, 3) if total else 0,
            "avg_frt_min": round(frt.mean(), 1) if len(frt) else None,
            "median_frt_min": round(frt.median(), 1) if len(frt) else None,
            "avg_resolution_min": round(res_time.mean(), 1) if len(res_time) else None,
            "avg_csat": round(csat.mean(), 2) if len(csat) else None,
            "avg_cost_usd": round(team_df["cost_usd"].mean(), 2),
            "total_cost_usd": round(team_df["cost_usd"].sum(), 2),
            "cost_per_resolved": round(cost_resolved / resolved, 2) if resolved else None,
        })

    return pd.DataFrame(records)


def compute_channel_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute performance metrics grouped by channel."""
    channels = df.groupby("channel")
    records = []

    for ch, ch_df in channels:
        total = len(ch_df)
        resolved = ch_df["is_resolved"].sum()
        frt = ch_df["first_response_min"].dropna()
        csat = ch_df.loc[ch_df["csat_score"].between(1, 5), "csat_score"]

        records.append({
            "channel": ch,
            "ticket_count": total,
            "resolution_rate": round(resolved / total, 3) if total else 0,
            "avg_frt_min": round(frt.mean(), 1) if len(frt) else None,
            "avg_csat": round(csat.mean(), 2) if len(csat) else None,
            "avg_cost_usd": round(ch_df["cost_usd"].mean(), 2),
            "total_cost_usd": round(ch_df["cost_usd"].sum(), 2),
        })

    return pd.DataFrame(records)


def compute_category_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute performance metrics grouped by category."""
    cats = df.groupby("category")
    records = []

    for cat, cat_df in cats:
        total = len(cat_df)
        resolved = cat_df["is_resolved"].sum()
        csat = cat_df.loc[cat_df["csat_score"].between(1, 5), "csat_score"]

        records.append({
            "category": cat,
            "ticket_count": total,
            "resolution_rate": round(resolved / total, 3) if total else 0,
            "escalation_rate": round((cat_df["resolution_status"] == "escalated").sum() / total, 3) if total else 0,
            "avg_csat": round(csat.mean(), 2) if len(csat) else None,
            "avg_cost_usd": round(cat_df["cost_usd"].mean(), 2),
            "total_cost_usd": round(cat_df["cost_usd"].sum(), 2),
        })

    return pd.DataFrame(records)


def find_statistical_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Identify statistical outliers in a numeric column.

    Args:
        df: Clean DataFrame.
        column: Numeric column to analyze.
        method: 'iqr' (interquartile range) or 'zscore'.
        threshold: IQR multiplier (default 1.5) or z-score threshold (default 3).

    Returns:
        DataFrame containing only the outlier rows.
    """
    values = df[column].dropna()

    if method == "iqr":
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = df[column].notna() & ((df[column] < lower) | (df[column] > upper))
    elif method == "zscore":
        mean = values.mean()
        std = values.std()
        if std == 0:
            return df.head(0)
        z_scores = (df[column] - mean) / std
        mask = df[column].notna() & (z_scores.abs() > threshold)
    else:
        raise ValueError(f"Unknown method: {method}")

    return df[mask]


def compute_chatbot_escalation_analysis(df: pd.DataFrame) -> dict:
    """Deep dive into chatbot escalation patterns.

    Returns dict with escalation analysis by category and subcategory.
    """
    bot_df = df[df["assigned_team"] == "ai_chatbot"]
    total_bot = len(bot_df)
    escalated_bot = bot_df[bot_df["resolution_status"] == "escalated"]

    by_category = (
        escalated_bot.groupby("category")
        .size()
        .reset_index(name="escalated_count")
    )
    by_category["total_in_category"] = bot_df.groupby("category").size().values
    by_category["escalation_rate"] = (
        by_category["escalated_count"] / by_category["total_in_category"]
    ).round(3)
    by_category = by_category.sort_values("escalation_rate", ascending=False)

    return {
        "total_chatbot_tickets": int(total_bot),
        "total_escalated": int(len(escalated_bot)),
        "overall_escalation_rate": round(len(escalated_bot) / total_bot, 3) if total_bot else 0,
        "by_category": by_category.to_dict("records"),
    }


def run_correlation_analysis(
    df: pd.DataFrame, target: str, features: list[str]
) -> dict:
    """Compute correlations between a target variable and feature variables.

    Args:
        df: Clean DataFrame.
        target: Target column name (e.g., 'csat_score').
        features: List of feature column names.

    Returns:
        Dict mapping feature name to correlation coefficient.
    """
    results = {}
    target_values = df[target].dropna()

    for feature in features:
        valid = df[[target, feature]].dropna()
        if len(valid) < 10:
            continue
        corr = valid[target].corr(valid[feature])
        results[feature] = round(corr, 3)

    return dict(sorted(results.items(), key=lambda x: abs(x[1]), reverse=True))
