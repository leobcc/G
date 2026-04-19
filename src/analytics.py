"""Statistical analysis functions for the Ops Intelligence system.

Each function takes a clean DataFrame and returns structured results.
All functions are pure — no side effects, no LLM calls.
"""

import logging

import pandas as pd

from src.config import COMPLETE_WEEKS

logger = logging.getLogger(__name__)


def compute_week_date_ranges(
    df: pd.DataFrame, complete_weeks: list[int] | None = None,
) -> dict[int, str]:
    """Map each complete week number to its human-readable date range.

    Args:
        df: Clean DataFrame with ``created_at`` and ``week_number`` columns.
        complete_weeks: Week numbers to include.  Defaults to ``COMPLETE_WEEKS``.

    Returns:
        Dict mapping week number to a string like ``"Feb 9 - Feb 15"``.
    """
    weeks = complete_weeks if complete_weeks is not None else COMPLETE_WEEKS
    weekly = df[df["week_number"].isin(weeks)]
    result: dict[int, str] = {}
    for wk, wdf in weekly.groupby("week_number"):
        start = wdf["created_at"].min()
        end = wdf["created_at"].max()
        try:
            # Windows uses %#d for non-padded day
            start_str = start.strftime("%b %#d")
            end_str = end.strftime("%b %#d")
        except ValueError:
            # Linux/macOS fallback
            start_str = start.strftime("%b %d").replace(" 0", " ")
            end_str = end.strftime("%b %d").replace(" 0", " ")
        result[int(wk)] = f"{start_str} - {end_str}"
    return result


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

    resolved = df["is_resolved"].sum() if "is_resolved" in df.columns else 0
    escalated = (df["resolution_status"] == "escalated").sum() if "resolution_status" in df.columns else 0
    abandoned = (df["resolution_status"] == "abandoned").sum() if "resolution_status" in df.columns else 0

    frt = df["first_response_min"].dropna() if "first_response_min" in df.columns else pd.Series(dtype=float)
    res = (
        df.loc[df["resolution_min"].notna() & (df["resolution_min"] >= 0), "resolution_min"]
        if "resolution_min" in df.columns else pd.Series(dtype=float)
    )
    csat = (
        df.loc[df["csat_score"].between(1, 5), "csat_score"]
        if "csat_score" in df.columns else pd.Series(dtype=float)
    )

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
        "avg_cost_usd": round(df["cost_usd"].mean(), 2) if "cost_usd" in df.columns else None,
        "total_cost_usd": round(df["cost_usd"].sum(), 2) if "cost_usd" in df.columns else None,
        "avg_contacts_per_ticket": round(df["contacts_per_ticket"].mean(), 1) if "contacts_per_ticket" in df.columns else None,
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
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    complete_weeks: list[int] | None = None,
) -> pd.DataFrame:
    """Compute key metrics for each complete week.

    If *metrics* is given they must be raw numeric column names suitable for
    ``.mean()`` aggregation.  When *metrics* is ``None``, a standard set of
    derived KPIs is computed: resolution_rate, escalation_rate,
    abandonment_rate, avg_csat, avg_frt, avg_resolution_min, avg_cost,
    ticket_count.

    Args:
        df: Clean DataFrame.
        metrics: Optional list of raw column names to aggregate.
        complete_weeks: Week numbers to include.  Defaults to ``COMPLETE_WEEKS``.

    Returns:
        DataFrame with weeks as rows and metric columns.
    """
    weeks = complete_weeks if complete_weeks is not None else COMPLETE_WEEKS
    weekly = df[df["week_number"].isin(weeks)]

    if metrics is not None:
        # Filter to only metrics that actually exist in the DataFrame
        available = [m for m in metrics if m in weekly.columns]
        if not available:
            return pd.DataFrame({"week_number": weeks})
        return weekly.groupby("week_number")[available].mean().reset_index()

    # Compute standard derived KPIs per week
    records = []
    for wk, wdf in weekly.groupby("week_number"):
        total = len(wdf)
        row: dict = {"week_number": wk, "ticket_count": total}
        if "is_resolved" in wdf.columns:
            row["resolution_rate"] = round(wdf["is_resolved"].mean(), 4)
        if "resolution_status" in wdf.columns:
            row["escalation_rate"] = round(
                (wdf["resolution_status"] == "escalated").mean(), 4
            )
            row["abandonment_rate"] = round(
                (wdf["resolution_status"] == "abandoned").mean(), 4
            )
        if "csat_score" in wdf.columns:
            valid_csat = wdf.loc[wdf["csat_score"].between(1, 5), "csat_score"]
            row["avg_csat"] = round(valid_csat.mean(), 3) if len(valid_csat) else None
        if "first_response_min" in wdf.columns:
            frt_vals = wdf["first_response_min"].dropna()
            row["avg_frt"] = round(frt_vals.mean(), 2) if len(frt_vals) else None
        if "resolution_min" in wdf.columns:
            res_vals = wdf.loc[wdf["resolution_min"] >= 0, "resolution_min"].dropna()
            row["avg_resolution_min"] = round(res_vals.mean(), 2) if len(res_vals) else None
        if "cost_usd" in wdf.columns:
            row["avg_cost"] = round(wdf["cost_usd"].mean(), 2)
            row["total_cost"] = round(wdf["cost_usd"].sum(), 2)
        records.append(row)
    return pd.DataFrame(records)


def compute_wow_kpis(
    df: pd.DataFrame, complete_weeks: list[int] | None = None,
) -> dict:
    """Compute week-over-week KPI deltas for the two most recent complete weeks.

    Uses the last available complete week as the *current* week and the
    week immediately before it as the *prior* week.  Incomplete weeks
    (those not in *complete_weeks*) are excluded.

    Args:
        df: Clean DataFrame.
        complete_weeks: Week numbers to consider.  Defaults to ``COMPLETE_WEEKS``.

    Returns:
        Dict with ``current_week``, ``prior_week``, ``current`` KPIs,
        ``prior`` KPIs, and ``deltas`` mapping each metric to its
        absolute change and direction emoji.
    """
    complete = sorted(complete_weeks if complete_weeks is not None else COMPLETE_WEEKS)
    if len(complete) < 2:
        return {"current_week": None, "prior_week": None, "deltas": {}}

    current_week = complete[-1]
    prior_week = complete[-2]

    current_kpi = compute_kpi_summary(df, week=current_week)
    prior_kpi = compute_kpi_summary(df, week=prior_week)

    # Metrics where lower = better
    lower_is_better = {
        "avg_first_response_min", "median_first_response_min",
        "avg_resolution_min", "median_resolution_min",
        "avg_cost_usd", "total_cost_usd",
        "avg_contacts_per_ticket",
        "escalation_rate", "abandonment_rate",
    }

    deltas: dict[str, dict] = {}
    for key in current_kpi:
        cur = current_kpi.get(key)
        pri = prior_kpi.get(key)
        if cur is None or pri is None:
            continue
        try:
            abs_change = cur - pri
            pct_change = (abs_change / abs(pri) * 100) if pri != 0 else 0.0
        except (TypeError, ZeroDivisionError):
            continue

        if key in lower_is_better:
            direction = "improving" if abs_change < 0 else ("worsening" if abs_change > 0 else "stable")
        else:
            direction = "improving" if abs_change > 0 else ("worsening" if abs_change < 0 else "stable")

        deltas[key] = {
            "abs_change": round(abs_change, 3),
            "pct_change": round(pct_change, 1),
            "direction": direction,
        }

    return {
        "current_week": current_week,
        "prior_week": prior_week,
        "current": current_kpi,
        "prior": prior_kpi,
        "deltas": deltas,
    }


def compute_team_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive team performance comparison.

    Returns DataFrame with one row per team and columns for each metric.
    """
    if "assigned_team" not in df.columns:
        return pd.DataFrame()

    teams = df.groupby("assigned_team")
    records = []

    for team_name, team_df in teams:
        total = len(team_df)
        row: dict = {"team": team_name, "ticket_count": total}

        has_resolved = "is_resolved" in team_df.columns
        has_status = "resolution_status" in team_df.columns
        has_frt = "first_response_min" in team_df.columns
        has_res = "resolution_min" in team_df.columns
        has_csat = "csat_score" in team_df.columns
        has_cost = "cost_usd" in team_df.columns

        resolved = int(team_df["is_resolved"].sum()) if has_resolved else 0
        row["resolution_rate"] = round(resolved / total, 3) if total else 0

        if has_status:
            row["escalation_rate"] = round((team_df["resolution_status"] == "escalated").sum() / total, 3) if total else 0
            row["abandonment_rate"] = round((team_df["resolution_status"] == "abandoned").sum() / total, 3) if total else 0

        if has_frt:
            frt = team_df["first_response_min"].dropna()
            row["avg_frt_min"] = round(frt.mean(), 1) if len(frt) else None
            row["median_frt_min"] = round(frt.median(), 1) if len(frt) else None

        if has_res:
            res_time = team_df.loc[team_df["resolution_min"].notna() & (team_df["resolution_min"] >= 0), "resolution_min"]
            row["avg_resolution_min"] = round(res_time.mean(), 1) if len(res_time) else None

        if has_csat:
            csat = team_df.loc[team_df["csat_score"].between(1, 5), "csat_score"]
            row["avg_csat"] = round(csat.mean(), 2) if len(csat) else None

        if has_cost:
            row["avg_cost_usd"] = round(team_df["cost_usd"].mean(), 2)
            row["total_cost_usd"] = round(team_df["cost_usd"].sum(), 2)
            if has_resolved and resolved:
                cost_resolved = team_df.loc[team_df["is_resolved"], "cost_usd"].sum()
                row["cost_per_resolved"] = round(cost_resolved / resolved, 2)

        records.append(row)

    return pd.DataFrame(records)


def compute_channel_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute performance metrics grouped by channel."""
    if "channel" not in df.columns:
        return pd.DataFrame()

    channels = df.groupby("channel")
    records = []

    for ch, ch_df in channels:
        total = len(ch_df)
        row: dict = {"channel": ch, "ticket_count": total}

        if "is_resolved" in ch_df.columns:
            resolved = int(ch_df["is_resolved"].sum())
            row["resolution_rate"] = round(resolved / total, 3) if total else 0
        if "first_response_min" in ch_df.columns:
            frt = ch_df["first_response_min"].dropna()
            row["avg_frt_min"] = round(frt.mean(), 1) if len(frt) else None
        if "csat_score" in ch_df.columns:
            csat = ch_df.loc[ch_df["csat_score"].between(1, 5), "csat_score"]
            row["avg_csat"] = round(csat.mean(), 2) if len(csat) else None
        if "cost_usd" in ch_df.columns:
            row["avg_cost_usd"] = round(ch_df["cost_usd"].mean(), 2)
            row["total_cost_usd"] = round(ch_df["cost_usd"].sum(), 2)

        records.append(row)

    return pd.DataFrame(records)


def compute_category_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute performance metrics grouped by category."""
    if "category" not in df.columns:
        return pd.DataFrame()

    cats = df.groupby("category")
    records = []

    for cat, cat_df in cats:
        total = len(cat_df)
        row: dict = {"category": cat, "ticket_count": total}

        if "is_resolved" in cat_df.columns:
            resolved = int(cat_df["is_resolved"].sum())
            row["resolution_rate"] = round(resolved / total, 3) if total else 0
        if "resolution_status" in cat_df.columns:
            row["escalation_rate"] = round((cat_df["resolution_status"] == "escalated").sum() / total, 3) if total else 0
        if "csat_score" in cat_df.columns:
            csat = cat_df.loc[cat_df["csat_score"].between(1, 5), "csat_score"]
            row["avg_csat"] = round(csat.mean(), 2) if len(csat) else None
        if "cost_usd" in cat_df.columns:
            row["avg_cost_usd"] = round(cat_df["cost_usd"].mean(), 2)
            row["total_cost_usd"] = round(cat_df["cost_usd"].sum(), 2)

        records.append(row)

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
    if "assigned_team" not in df.columns:
        return {"total_chatbot_tickets": 0, "total_escalated": 0,
                "overall_escalation_rate": 0, "by_category": []}

    bot_df = df[df["assigned_team"] == "ai_chatbot"]
    total_bot = len(bot_df)
    if total_bot == 0:
        return {"total_chatbot_tickets": 0, "total_escalated": 0,
                "overall_escalation_rate": 0, "by_category": []}

    escalated_bot = bot_df[bot_df["resolution_status"] == "escalated"]

    # Use merge instead of .values alignment to avoid mismatched arrays
    esc_by_cat = (
        escalated_bot.groupby("category")
        .size()
        .reset_index(name="escalated_count")
    )
    total_by_cat = (
        bot_df.groupby("category")
        .size()
        .reset_index(name="total_in_category")
    )
    by_category = total_by_cat.merge(esc_by_cat, on="category", how="left")
    by_category["escalated_count"] = by_category["escalated_count"].fillna(0).astype(int)
    by_category["escalation_rate"] = (
        by_category["escalated_count"] / by_category["total_in_category"]
    ).round(3)
    by_category = by_category.sort_values("escalation_rate", ascending=False)

    return {
        "total_chatbot_tickets": int(total_bot),
        "total_escalated": int(len(escalated_bot)),
        "overall_escalation_rate": round(len(escalated_bot) / total_bot, 3),
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
        Dict mapping feature name to {correlation, p_value, n} dict.
    """
    from scipy import stats

    results = {}

    for feature in features:
        valid = df[[target, feature]].dropna()
        if len(valid) < 10:
            continue
        corr, p_val = stats.pearsonr(valid[target], valid[feature])
        results[feature] = {
            "correlation": round(corr, 3),
            "p_value": round(p_val, 6),
            "n": len(valid),
        }

    return dict(sorted(results.items(), key=lambda x: abs(x[1]["correlation"]), reverse=True))
