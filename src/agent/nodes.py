"""LangGraph node functions.

Each function is a node in the analytical pipeline graph.
Nodes read from AgentState, perform work, and return state updates.

Because ``execution_log`` and ``errors`` use an ``operator.add`` reducer,
each node returns a *list* of new entries and LangGraph concatenates them
automatically — no need to read the existing list.
"""

import json
import logging
import re
from io import StringIO

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.prompts import (
    OPPORTUNITY_SCORING_PROMPT,
    REPORT_GENERATION_PROMPT,
)
from src.agent.state import AgentState
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
from src.config import LLM_MODEL_ANALYSIS, LLM_TEMPERATURE, RAW_DATA_PATH
from src.data_cleaning import clean_data, load_raw_data

logger = logging.getLogger(__name__)


def _get_llm(model: str | None = None) -> ChatGoogleGenerativeAI:
    """Create a Google Gemini LLM instance (free tier)."""
    return ChatGoogleGenerativeAI(
        model=model or LLM_MODEL_ANALYSIS,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=8192,
    )


# ---------------------------------------------------------------------------
# Node 1: Data Ingestion
# ---------------------------------------------------------------------------
def node_ingest_data(state: AgentState) -> dict:
    """Load raw data from CSV and store path + row count."""
    logger.info("Node: ingest_data -- loading CSV")
    raw_df = load_raw_data()
    return {
        "raw_df_path": str(RAW_DATA_PATH),
        "raw_row_count": len(raw_df),
        "current_step": "data_ingested",
        "execution_log": [f"Loaded {len(raw_df):,} rows from CSV"],
    }


# ---------------------------------------------------------------------------
# Node 2: Data Quality
# ---------------------------------------------------------------------------
def node_data_quality(state: AgentState) -> dict:
    """Clean data and run quality checks."""
    logger.info("Node: data_quality -- cleaning data")
    raw_df = load_raw_data()
    clean_df, cleaning_log = clean_data(raw_df)
    quality = json.loads(tool_data_quality(raw_df, clean_df))
    quality["cleaning_log"] = cleaning_log

    return {
        "clean_df_serialized": clean_df.to_json(orient="records", date_format="iso"),
        "data_quality": quality,
        "current_step": "data_cleaned",
        "execution_log": [
            f"Cleaned data: fixed {sum(cleaning_log.values())} issues "
            f"({', '.join(f'{k}={v}' for k, v in cleaning_log.items())})"
        ],
    }


# ---------------------------------------------------------------------------
# Node 3a: Trend Analysis
# ---------------------------------------------------------------------------
def node_trend_analysis(state: AgentState) -> dict:
    """Compute all trend and performance metrics."""
    logger.info("Node: trend_analysis")
    df = pd.read_json(StringIO(state["clean_df_serialized"]))

    kpi = json.loads(tool_compute_kpis(df))
    teams = json.loads(tool_team_performance(df))
    channels = json.loads(tool_channel_performance(df))
    categories = json.loads(tool_category_performance(df))
    trends = json.loads(tool_weekly_trends(df))
    correlations = json.loads(tool_correlation_analysis(df))
    kpi["correlations"] = correlations

    return {
        "kpi_summary": kpi,
        "weekly_trends": trends,
        "team_performance": teams,
        "channel_performance": channels,
        "category_performance": categories,
        "execution_log": ["Trend analysis complete"],
    }


# ---------------------------------------------------------------------------
# Node 3b: Anomaly Detection
# ---------------------------------------------------------------------------
def node_anomaly_detection(state: AgentState) -> dict:
    """Detect anomalies and analyze chatbot escalations."""
    logger.info("Node: anomaly_detection")
    df = pd.read_json(StringIO(state["clean_df_serialized"]))

    anomalies = {}
    for col in [
        "first_response_min",
        "resolution_min",
        "cost_usd",
        "contacts_per_ticket",
    ]:
        anomalies[col] = json.loads(tool_find_anomalies(df, col))

    chatbot = json.loads(tool_chatbot_escalation(df))

    total_outliers = sum(v["total_outliers"] for v in anomalies.values())
    return {
        "anomalies": anomalies,
        "chatbot_escalation": chatbot,
        "execution_log": [
            f"Anomaly detection complete: {total_outliers} outliers, "
            f"chatbot escalation rate {chatbot.get('overall_escalation_rate', 'N/A')}"
        ],
    }


# ---------------------------------------------------------------------------
# Node 3c: NLP Analysis
# ---------------------------------------------------------------------------
def node_nlp_analysis(state: AgentState) -> dict:
    """Run NLP pipeline on ticket text."""
    logger.info("Node: nlp_analysis")
    df = pd.read_json(StringIO(state["clean_df_serialized"]))
    nlp = json.loads(tool_nlp_analysis(df))

    return {
        "nlp_summary": nlp,
        "execution_log": [
            f"NLP analysis complete: frustration rate {nlp.get('frustration_rate', 'N/A')}, "
            f"avg sentiment {nlp.get('avg_sentiment_polarity', 'N/A')}"
        ],
    }


# ---------------------------------------------------------------------------
# Node 4: Opportunity Scoring (LLM-powered)
# ---------------------------------------------------------------------------
def node_opportunity_scoring(state: AgentState) -> dict:
    """Use LLM to identify and score improvement opportunities."""
    logger.info("Node: opportunity_scoring")
    llm = _get_llm(LLM_MODEL_ANALYSIS)

    # Build a concise context for the LLM
    context = json.dumps(
        {
            "kpis": state.get("kpi_summary", {}),
            "team_performance": state.get("team_performance", []),
            "chatbot_escalation": state.get("chatbot_escalation", {}),
            "nlp_summary": {
                k: v
                for k, v in state.get("nlp_summary", {}).items()
                if k != "topics"  # topics can be large; omit for token saving
            },
            "anomalies": {
                k: {"total_outliers": v.get("total_outliers", 0)}
                for k, v in state.get("anomalies", {}).items()
            },
            "weekly_trends": state.get("weekly_trends", []),
        },
        indent=2,
        default=str,
    )

    messages = [
        SystemMessage(content=OPPORTUNITY_SCORING_PROMPT),
        HumanMessage(content=f"Here is the analytical context:\n\n{context}"),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content

        # Strip markdown code fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content.strip())
        content = re.sub(r"\s*```$", "", content.strip())

        opportunities = json.loads(content)
        if isinstance(opportunities, dict) and "opportunities" in opportunities:
            opportunities = opportunities["opportunities"]
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM opportunity response as JSON")
        opportunities = []
    except Exception as e:
        logger.error("LLM call failed in opportunity_scoring: %s", e)
        opportunities = []

    return {
        "opportunities": opportunities,
        "current_step": "opportunities_scored",
        "execution_log": [f"Identified {len(opportunities)} opportunities"],
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
            "kpis": state.get("kpi_summary", {}),
            "team_performance": state.get("team_performance", []),
            "chatbot_escalation": state.get("chatbot_escalation", {}),
            "nlp_summary": {
                k: v
                for k, v in state.get("nlp_summary", {}).items()
                if k != "topics"
            },
            "opportunities": state.get("opportunities", []),
            "anomalies": {
                k: {"total_outliers": v.get("total_outliers", 0)}
                for k, v in state.get("anomalies", {}).items()
            },
            "data_quality": {
                "total_rows": state.get("data_quality", {}).get("total_rows"),
                "completeness_score": state.get("data_quality", {}).get(
                    "completeness_score"
                ),
                "cleaning_log": state.get("data_quality", {}).get("cleaning_log"),
            },
            "weekly_trends": state.get("weekly_trends", []),
        },
        indent=2,
        default=str,
    )

    try:
        messages = [
            SystemMessage(content=REPORT_GENERATION_PROMPT),
            HumanMessage(
                content=f"Generate the weekly brief from this data:\n\n{context}"
            ),
        ]
        response = llm.invoke(messages)
        report = response.content
    except Exception as e:
        logger.error("LLM call failed in report_generation: %s", e)
        report = f"# Report Generation Failed\n\nError: {e}"

    return {
        "report_markdown": report,
        "current_step": "report_generated",
        "execution_log": ["Weekly brief generated"],
    }
