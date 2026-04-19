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
import time
from io import StringIO

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.prompts import (
    EXECUTIVE_INSIGHTS_PROMPT,
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
from src.analytics import compute_week_date_ranges, compute_wow_kpis
from src.config import COMPLETE_WEEKS, LLM_MODEL_ANALYSIS, LLM_TEMPERATURE, RAW_DATA_PATH
from src.data_cleaning import clean_data, detect_complete_weeks, load_raw_data

logger = logging.getLogger(__name__)


# Maximum retries for LLM calls on rate-limit errors
_MAX_RETRIES = 4
_INITIAL_BACKOFF_S = 10.0


def _get_llm(model: str | None = None) -> ChatGroq:
    """Create a Groq LLM instance (free tier, LPU inference)."""
    return ChatGroq(
        model=model or LLM_MODEL_ANALYSIS,
        temperature=LLM_TEMPERATURE,
        max_tokens=8192,
    )


def _invoke_with_retry(llm: ChatGroq, messages: list, *, label: str = "LLM") -> str:
    """Invoke the LLM with exponential backoff on 429 / rate-limit errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate" in err_str.lower()
            if is_rate_limit and attempt < _MAX_RETRIES - 1:
                wait = _INITIAL_BACKOFF_S * (2 ** attempt)
                logger.warning("%s rate-limited (attempt %d/%d), retrying in %.0fs ...", label, attempt + 1, _MAX_RETRIES, wait)
                time.sleep(wait)
            else:
                raise
    # unreachable
    raise RuntimeError(f"{label}: exhausted retries")


# ---------------------------------------------------------------------------
# Node 1: Data Ingestion
# ---------------------------------------------------------------------------
def node_ingest_data(state: AgentState) -> dict:
    """Load raw data from CSV and store path + row count."""
    logger.info("Node: ingest_data -- loading CSV")
    path = state.get("raw_df_path") or str(RAW_DATA_PATH)
    raw_df = load_raw_data(path)
    return {
        "raw_df_path": path,
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
    path = state.get("raw_df_path") or str(RAW_DATA_PATH)
    raw_df = load_raw_data(path)
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

    # Dynamically detect complete weeks from the data
    complete_weeks = detect_complete_weeks(df)
    if not complete_weeks:
        # Fallback: use config defaults (never blindly use all weeks)
        complete_weeks = list(COMPLETE_WEEKS)
    complete_df = df[df["week_number"].isin(complete_weeks)] if complete_weeks else df

    kpi = json.loads(tool_compute_kpis(complete_df))
    teams = json.loads(tool_team_performance(complete_df))
    channels = json.loads(tool_channel_performance(complete_df))
    categories = json.loads(tool_category_performance(complete_df))
    trends = json.loads(tool_weekly_trends(df))
    correlations = json.loads(tool_correlation_analysis(complete_df))
    kpi["correlations"] = correlations

    # Compute WoW deltas and week date ranges for LLM context
    wow = compute_wow_kpis(df, complete_weeks=complete_weeks)
    date_ranges = compute_week_date_ranges(df, complete_weeks=complete_weeks)

    return {
        "kpi_summary": kpi,
        "weekly_trends": trends,
        "team_performance": teams,
        "channel_performance": channels,
        "category_performance": categories,
        "wow_kpis": wow,
        "week_date_ranges": date_ranges,
        "complete_weeks": complete_weeks,
        "execution_log": [f"Trend analysis complete ({len(complete_weeks)} complete weeks)"],
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

    # Build WoW summary for temporal context
    wow = state.get("wow_kpis", {})
    date_ranges = state.get("week_date_ranges", {})
    _cw = state.get("complete_weeks", COMPLETE_WEEKS)
    cur_wk = wow.get("current_week")
    pri_wk = wow.get("prior_week")

    temporal_context = {
        "analysis_period": f"Weeks {min(_cw)}-{max(_cw)}" if _cw else "All weeks",
        "week_date_ranges": date_ranges,
        "current_week": cur_wk,
        "prior_week": pri_wk,
        "current_week_kpis": wow.get("current", {}),
        "prior_week_kpis": wow.get("prior", {}),
        "wow_deltas": wow.get("deltas", {}),
    }

    # Build a concise context for the LLM
    context = json.dumps(
        {
            "temporal_context": temporal_context,
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
        content = _invoke_with_retry(llm, messages, label="opportunity_scoring")

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

    # Build WoW summary for temporal context
    wow = state.get("wow_kpis", {})
    date_ranges = state.get("week_date_ranges", {})
    _cw = state.get("complete_weeks", COMPLETE_WEEKS)
    cur_wk = wow.get("current_week")
    pri_wk = wow.get("prior_week")

    temporal_context = {
        "analysis_period": f"Weeks {min(_cw)}-{max(_cw)}" if _cw else "All weeks",
        "week_date_ranges": date_ranges,
        "current_week": cur_wk,
        "current_week_dates": date_ranges.get(cur_wk, ""),
        "prior_week": pri_wk,
        "prior_week_dates": date_ranges.get(pri_wk, ""),
        "current_week_kpis": wow.get("current", {}),
        "prior_week_kpis": wow.get("prior", {}),
        "wow_deltas": wow.get("deltas", {}),
    }

    # Build a COMPACT context to stay within Groq free-tier TPM limits.
    # Only include top-level KPIs, team names + CSAT, and opportunity titles.
    kpi = state.get("kpi_summary", {})
    compact_kpi = {
        k: kpi[k]
        for k in [
            "total_tickets", "resolution_rate", "escalation_rate",
            "abandonment_rate", "avg_first_response_min", "avg_resolution_min",
            "avg_csat", "avg_cost_usd", "total_cost_usd",
        ]
        if k in kpi
    }

    compact_teams = [
        {
            "team": t.get("team"),
            "avg_csat": t.get("avg_csat"),
            "resolution_rate": t.get("resolution_rate"),
            "avg_cost_usd": t.get("avg_cost_usd"),
        }
        for t in state.get("team_performance", [])
    ]

    compact_opps = [
        {"rank": o.get("rank", i + 1), "title": o.get("title", "")}
        for i, o in enumerate(state.get("opportunities", []))
    ]

    nlp = state.get("nlp_summary", {})
    compact_nlp = {
        "avg_polarity": nlp.get("avg_sentiment_polarity"),
        "frustration_rate": nlp.get("frustration_rate"),
        "sentiment_dist": nlp.get("sentiment_distribution"),
    }

    chatbot = state.get("chatbot_escalation", {})
    compact_chatbot = {
        "escalation_rate": chatbot.get("overall_escalation_rate"),
    }

    context = json.dumps(
        {
            "temporal_context": temporal_context,
            "kpis": compact_kpi,
            "teams": compact_teams,
            "chatbot": compact_chatbot,
            "nlp": compact_nlp,
            "opportunities": compact_opps,
            "weekly_trends": state.get("weekly_trends", []),
            "quality": {
                "rows": state.get("data_quality", {}).get("total_rows"),
                "completeness": state.get("data_quality", {}).get("completeness_score"),
            },
        },
        indent=1,
        default=str,
    )

    try:
        # Brief pause between LLM calls to stay within RPM limits
        time.sleep(2)

        # Interpolate week info into the prompt
        prompt = REPORT_GENERATION_PROMPT.replace(
            "{week}",
            f"{cur_wk} ({date_ranges.get(cur_wk, '')})" if cur_wk else "N/A",
        )

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(
                content=f"Generate the weekly brief from this data:\n\n{context}"
            ),
        ]
        report = _invoke_with_retry(llm, messages, label="report_generation")
    except Exception as e:
        logger.error("LLM call failed in report_generation: %s", e)
        report = f"# Report Generation Failed\n\nError: {e}"

    return {
        "report_markdown": report,
        "execution_log": ["Weekly brief generated"],
    }


# ---------------------------------------------------------------------------
# Node 6: Executive Insights (LLM-powered)
# ---------------------------------------------------------------------------
def node_executive_insights(state: AgentState) -> dict:
    """Generate plain-language executive insights for dashboard sections."""
    logger.info("Node: executive_insights")
    llm = _get_llm(LLM_MODEL_ANALYSIS)

    # Build WoW summary for temporal context
    wow = state.get("wow_kpis", {})
    date_ranges = state.get("week_date_ranges", {})
    _cw = state.get("complete_weeks", COMPLETE_WEEKS)
    cur_wk = wow.get("current_week")
    pri_wk = wow.get("prior_week")

    temporal_context = {
        "analysis_period": f"Weeks {min(_cw)}-{max(_cw)}" if _cw else "All weeks",
        "week_date_ranges": date_ranges,
        "current_week": cur_wk,
        "current_week_dates": date_ranges.get(cur_wk, ""),
        "prior_week": pri_wk,
        "prior_week_dates": date_ranges.get(pri_wk, ""),
        "current_week_kpis": wow.get("current", {}),
        "prior_week_kpis": wow.get("prior", {}),
        "wow_deltas": wow.get("deltas", {}),
    }

    # Compact context — avoid exceeding Groq free-tier TPM
    kpi = state.get("kpi_summary", {})
    compact_kpi = {
        k: kpi[k]
        for k in [
            "total_tickets", "resolution_rate", "escalation_rate",
            "abandonment_rate", "avg_first_response_min", "avg_csat",
            "avg_cost_usd",
        ]
        if k in kpi
    }

    compact_teams = [
        {
            "team": t.get("team"),
            "avg_csat": t.get("avg_csat"),
            "resolution_rate": t.get("resolution_rate"),
        }
        for t in state.get("team_performance", [])
    ]

    nlp = state.get("nlp_summary", {})
    compact_nlp = {
        "avg_polarity": nlp.get("avg_sentiment_polarity"),
        "frustration_rate": nlp.get("frustration_rate"),
        "sentiment_dist": nlp.get("sentiment_distribution"),
        "frustration_by_category": nlp.get("frustration_by_category"),
    }

    context = json.dumps(
        {
            "temporal_context": temporal_context,
            "overall_kpis": compact_kpi,
            "teams": compact_teams,
            "nlp": compact_nlp,
            "chatbot_escalation_rate": state.get("chatbot_escalation", {}).get(
                "overall_escalation_rate"
            ),
            "weekly_trends": state.get("weekly_trends", []),
        },
        indent=1,
        default=str,
    )

    default_insights = {
        "trends_insight": "",
        "correlation_insight": "",
        "sentiment_insight": "",
        "frustration_insight": "",
        "topic_insight": "",
        "team_performance_insight": "",
        "sentiment_dimension_insight": "",
        "opportunities_intro": "",
    }

    try:
        time.sleep(2)
        messages = [
            SystemMessage(content=EXECUTIVE_INSIGHTS_PROMPT),
            HumanMessage(content=f"Generate executive insights from:\n\n{context}"),
        ]
        content = _invoke_with_retry(llm, messages, label="executive_insights")

        content = re.sub(r"^```(?:json)?\s*", "", content.strip())
        content = re.sub(r"\s*```$", "", content.strip())

        insights = json.loads(content)
        if not isinstance(insights, dict):
            insights = default_insights
    except json.JSONDecodeError:
        logger.warning("Failed to parse executive insights JSON")
        insights = default_insights
    except Exception as e:
        logger.error("LLM call failed in executive_insights: %s", e)
        insights = default_insights

    return {
        "executive_insights": insights,
        "execution_log": ["Executive insights generated"],
    }
