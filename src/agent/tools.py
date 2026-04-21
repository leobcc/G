"""LangGraph tool definitions.

Each tool wraps an analytical function for use by LLM nodes.
Tools are functions that the agent can call to get data.

Two levels:
1. Internal tool functions (tool_*) - take DataFrame, return JSON string
2. LLM-callable tool definitions (get_*_tool) - for @tool decorator
"""

import json
import logging
from typing import Any

import pandas as pd
from langchain_core.tools import tool

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


# ── Internal Tool Functions (take DataFrame, return JSON) ──────────────────


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


# ── LLM-Callable Tools (no DataFrame param, use @tool decorator) ──────────
# These are meant to be called by the LLM. The agent node will execute them
# using tool_executor_with_context() which provides the DataFrame.


@tool
def investigate_chatbot_performance() -> str:
    """Investigate chatbot escalation rates and performance issues.
    
    Returns analysis of chatbot escalations by category and overall metrics.
    Use when: you see high escalation rates or want to understand chatbot effectiveness.
    """
    return "PLACEHOLDER"  # Executor will fill this


@tool
def analyze_team_performance() -> str:
    """Analyze and compare performance across all teams (in-house, BPO Vendor A/B, chatbot).
    
    Returns CSAT, resolution rates, costs, and efficiency metrics per team.
    Use when: you want to identify team-specific performance gaps or leaders.
    """
    return "PLACEHOLDER"


@tool
def check_category_issues() -> str:
    """Deep dive into performance by ticket category (refund, billing, merchant_issue, etc.).
    
    Returns performance metrics, cost drivers, and CSAT by category.
    Use when: you see issues in specific categories or want to identify category bottlenecks.
    """
    return "PLACEHOLDER"


@tool
def find_cost_outliers() -> str:
    """Identify tickets and scenarios with abnormal costs or resolution times.
    
    Returns outlier count and samples of high-cost, slow-to-resolve tickets.
    Use when: investigating efficiency gaps or unusual cost spikes.
    """
    return "PLACEHOLDER"


@tool
def verify_kpis_with_trends() -> str:
    """Compare current week KPIs against historical trends and prior week.
    
    Returns week-over-week deltas and trend direction (improving/worsening/stable).
    Use when: validating improvement opportunities or checking if findings are significant.
    """
    return "PLACEHOLDER"


@tool
def verify_with_correlation_analysis() -> str:
    """Run correlation analysis to verify hypotheses about what drives CSAT or costs.
    
    Returns Pearson correlations between KPIs and factors like FRT, resolution time, etc.
    Use when: confirming causal relationships or validating the root cause of a problem.
    """
    return "PLACEHOLDER"


# ── Tool Executor (dispatches tool calls with DataFrame in context) ────────


def build_tool_executor(df: pd.DataFrame) -> dict[str, callable]:
    """Build a dict of tool name → executor function, binding DataFrame via closure.
    
    This creates a callable for each LLM tool that has access to the DataFrame.
    
    Args:
        df: The clean DataFrame for all analyses
        
    Returns:
        Dict mapping tool_name (from @tool decorated functions) to executor callable
    """
    executors = {
        "investigate_chatbot_performance": lambda: tool_chatbot_escalation(df),
        "analyze_team_performance": lambda: tool_team_performance(df),
        "check_category_issues": lambda: tool_category_performance(df),
        "find_cost_outliers": lambda: tool_find_anomalies(df, "cost_usd"),
        "verify_kpis_with_trends": lambda: tool_weekly_trends(df),
        "verify_with_correlation_analysis": lambda: tool_correlation_analysis(df),
    }
    return executors
