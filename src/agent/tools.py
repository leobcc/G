"""LangGraph tool definitions.

Each tool wraps an analytical function for use by LLM nodes.
Tools are functions that the agent can call to get data.
"""

import json
import logging

import pandas as pd

from src.analytics import (
    compute_category_performance,
    compute_channel_performance,
    compute_chatbot_escalation_analysis,
    compute_kpi_summary,
    compute_team_performance,
    compute_weekly_trends,
    find_statistical_outliers,
    run_correlation_analysis,
)
from src.data_cleaning import get_data_quality_report
from src.nlp_analysis import compute_nlp_summary

logger = logging.getLogger(__name__)


def tool_compute_kpis(df: pd.DataFrame, week: int | None = None) -> str:
    """Compute KPI summary and return as JSON string."""
    result = compute_kpi_summary(df, week=week)
    return json.dumps(result, indent=2)


def tool_team_performance(df: pd.DataFrame) -> str:
    """Compute team performance comparison and return as JSON string."""
    result = compute_team_performance(df)
    return result.to_json(orient="records", indent=2)


def tool_channel_performance(df: pd.DataFrame) -> str:
    """Compute channel performance and return as JSON string."""
    result = compute_channel_performance(df)
    return result.to_json(orient="records", indent=2)


def tool_category_performance(df: pd.DataFrame) -> str:
    """Compute category performance and return as JSON string."""
    result = compute_category_performance(df)
    return result.to_json(orient="records", indent=2)


def tool_weekly_trends(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
) -> str:
    """Compute weekly trends for key metrics."""
    result = compute_weekly_trends(df, metrics)
    return result.to_json(orient="records", indent=2)


def tool_find_anomalies(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 1.5,
) -> str:
    """Find outliers in a numeric column."""
    result = find_statistical_outliers(df, column, method, threshold)
    return json.dumps(
        {
            "total_outliers": len(result),
            "column": column,
            "method": method,
            "threshold": threshold,
            "sample": result.head(10).to_dict("records"),
        },
        indent=2,
        default=str,
    )


def tool_chatbot_escalation(df: pd.DataFrame) -> str:
    """Analyze chatbot escalation patterns."""
    result = compute_chatbot_escalation_analysis(df)
    return json.dumps(result, indent=2)


def tool_nlp_analysis(df: pd.DataFrame) -> str:
    """Run full NLP pipeline."""
    result = compute_nlp_summary(df)
    return json.dumps(result, indent=2, default=str)


def tool_correlation_analysis(
    df: pd.DataFrame,
    target: str = "csat_score",
    features: list[str] | None = None,
) -> str:
    """Run correlation analysis between target and feature variables."""
    if features is None:
        features = [
            "first_response_min",
            "resolution_min",
            "cost_usd",
            "contacts_per_ticket",
        ]
    result = run_correlation_analysis(df, target, features)
    return json.dumps(result, indent=2)


def tool_data_quality(
    raw_df: pd.DataFrame, clean_df: pd.DataFrame
) -> str:
    """Generate data quality report."""
    result = get_data_quality_report(raw_df, clean_df)
    return json.dumps(result, indent=2, default=str)
