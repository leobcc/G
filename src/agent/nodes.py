"""LangGraph node functions.

Each function is a node in the analytical pipeline graph.
Nodes read from AgentState, perform work, and return state updates.
"""

import json
import logging
from typing import Any

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.prompts import (
    OPPORTUNITY_SCORING_PROMPT,
    REPORT_GENERATION_PROMPT,
    TREND_INTERPRETATION_PROMPT,
)
from src.agent.state import AgentState, OpportunityItem
from src.agent.tools import (
    tool_category_performance,
    tool_channel_performance,
    tool_chatbot_escalation,
    tool_compute_kpis,
    tool_correlation_analysis,
    tool_data_quality,
    tool_find_anomalies,
    tool_nlp_analysis,
    tool_team_performance,
    tool_weekly_trends,
)
from src.config import LLM_MODEL_ANALYSIS, LLM_MODEL_SIMPLE, LLM_TEMPERATURE
from src.data_cleaning import clean_data, load_raw_data

logger = logging.getLogger(__name__)


def _get_llm(model: str | None = None) -> ChatGoogleGenerativeAI:
    """Create a Google Gemini LLM instance (free tier)."""
    return ChatGoogleGenerativeAI(
        model=model or LLM_MODEL_ANALYSIS,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=4096,
    )


# ---------------------------------------------------------------------------
# Node 1: Data Ingestion
# ---------------------------------------------------------------------------
def node_ingest_data(state: AgentState) -> dict:
    """Load raw data from CSV."""
    logger.info("Node: ingest_data — loading CSV")
    raw_df = load_raw_data()
    return {
        "raw_df_path": str(raw_df),
        "raw_row_count": len(raw_df),
        "current_step": "data_ingested",
        "execution_log": state.get("execution_log", [])
        + [f"Loaded {len(raw_df)} rows"],
    }


# ---------------------------------------------------------------------------
# Node 2: Data Quality
# ---------------------------------------------------------------------------
def node_data_quality(state: AgentState) -> dict:
    """Clean data and run quality checks."""
    logger.info("Node: data_quality — cleaning data")
    raw_df = load_raw_data()
    clean_df, cleaning_log = clean_data(raw_df)
    quality = json.loads(tool_data_quality(raw_df, clean_df))
    quality["cleaning_log"] = cleaning_log

    return {
        "clean_df_serialized": clean_df.to_json(orient="records"),
        "data_quality": quality,
        "current_step": "data_cleaned",
        "execution_log": state.get("execution_log", [])
        + [f"Cleaned data, fixed {sum(cleaning_log.values())} issues"],
    }


# ---------------------------------------------------------------------------
# Node 3a: Trend Analysis
# ---------------------------------------------------------------------------
def node_trend_analysis(state: AgentState) -> dict:
    """Compute all trend and performance metrics."""
    logger.info("Node: trend_analysis")
    df = pd.read_json(state["clean_df_serialized"])

    kpi = json.loads(tool_compute_kpis(df))
    teams = json.loads(tool_team_performance(df))
    channels = json.loads(tool_channel_performance(df))
    categories = json.loads(tool_category_performance(df))
    trends = json.loads(tool_weekly_trends(df))
    correlations = json.loads(tool_correlation_analysis(df))

    return {
        "kpi_summary": kpi,
        "weekly_trends": trends,
        "team_performance": teams,
        "channel_performance": channels,
        "category_performance": categories,
        "current_step": "trends_analyzed",
        "execution_log": state.get("execution_log", [])
        + ["Trend analysis complete"],
    }


# ---------------------------------------------------------------------------
# Node 3b: Anomaly Detection
# ---------------------------------------------------------------------------
def node_anomaly_detection(state: AgentState) -> dict:
    """Detect anomalies and analyze chatbot escalations."""
    logger.info("Node: anomaly_detection")
    df = pd.read_json(state["clean_df_serialized"])

    anomalies = {}
    for col in ["first_response_min", "resolution_min", "cost_usd", "contacts_per_ticket"]:
        anomalies[col] = json.loads(tool_find_anomalies(df, col))

    chatbot = json.loads(tool_chatbot_escalation(df))

    return {
        "anomalies": anomalies,
        "chatbot_escalation": chatbot,
        "current_step": "anomalies_detected",
        "execution_log": state.get("execution_log", [])
        + ["Anomaly detection complete"],
    }


# ---------------------------------------------------------------------------
# Node 3c: NLP Analysis
# ---------------------------------------------------------------------------
def node_nlp_analysis(state: AgentState) -> dict:
    """Run NLP pipeline on ticket text."""
    logger.info("Node: nlp_analysis")
    df = pd.read_json(state["clean_df_serialized"])
    nlp = json.loads(tool_nlp_analysis(df))

    return {
        "nlp_summary": nlp,
        "current_step": "nlp_analyzed",
        "execution_log": state.get("execution_log", [])
        + ["NLP analysis complete"],
    }


# ---------------------------------------------------------------------------
# Node 4: Opportunity Scoring (LLM-powered)
# ---------------------------------------------------------------------------
def node_opportunity_scoring(state: AgentState) -> dict:
    """Use LLM to identify and score improvement opportunities."""
    logger.info("Node: opportunity_scoring")
    llm = _get_llm(LLM_MODEL_ANALYSIS)

    context = json.dumps(
        {
            "kpis": state["kpi_summary"],
            "team_performance": state["team_performance"],
            "chatbot_escalation": state["chatbot_escalation"],
            "nlp_summary": state["nlp_summary"],
            "anomalies": {
                k: {"total_outliers": v["total_outliers"]}
                for k, v in state["anomalies"].items()
            },
        },
        indent=2,
        default=str,
    )

    messages = [
        SystemMessage(content=OPPORTUNITY_SCORING_PROMPT),
        HumanMessage(content=f"Here is the analytical context:\n\n{context}"),
    ]

    response = llm.invoke(messages)

    # Parse LLM response into structured opportunities
    try:
        opportunities = json.loads(response.content)
        if isinstance(opportunities, dict) and "opportunities" in opportunities:
            opportunities = opportunities["opportunities"]
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM opportunity response as JSON")
        opportunities = []

    return {
        "opportunities": opportunities,
        "current_step": "opportunities_scored",
        "execution_log": state.get("execution_log", [])
        + [f"Identified {len(opportunities)} opportunities"],
    }


# ---------------------------------------------------------------------------
# Node 5: Report Generation (LLM-powered)
# ---------------------------------------------------------------------------
def node_report_generation(state: AgentState) -> dict:
    """Generate executive weekly brief using LLM."""
    logger.info("Node: report_generation")
    llm = _get_llm(LLM_MODEL_ANALYSIS)

    context = json.dumps(
        {
            "kpis": state["kpi_summary"],
            "team_performance": state["team_performance"],
            "chatbot_escalation": state["chatbot_escalation"],
            "nlp_summary": state.get("nlp_summary", {}),
            "opportunities": state.get("opportunities", []),
            "anomalies": {
                k: {"total_outliers": v["total_outliers"]}
                for k, v in state.get("anomalies", {}).items()
            },
        },
        indent=2,
        default=str,
    )

    messages = [
        SystemMessage(content=REPORT_GENERATION_PROMPT),
        HumanMessage(content=f"Generate the weekly brief from this data:\n\n{context}"),
    ]

    response = llm.invoke(messages)

    return {
        "report_markdown": response.content,
        "current_step": "report_generated",
        "execution_log": state.get("execution_log", [])
        + ["Weekly brief generated"],
    }
