"""LangGraph graph definition.

Defines the analytical pipeline as a directed graph:
  ingest → quality → [trends, anomalies, nlp] → opportunities → report
"""

import logging

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    node_anomaly_detection,
    node_data_quality,
    node_executive_insights,
    node_ingest_data,
    node_nlp_analysis,
    node_opportunity_scoring,
    node_report_generation,
    node_trend_analysis,
)
from src.agent.state import AgentState

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Build and compile the LangGraph analytical pipeline.

    Pipeline structure:
        ingest_data
            ↓
        data_quality
            ↓
        ┌────────────────┬──────────────────┐
        trend_analysis   anomaly_detection   nlp_analysis
        └────────────────┴──────────────────┘
            ↓
        opportunity_scoring
            ↓
        report_generation
            ↓
        executive_insights
            ↓
           END

    Returns:
        Compiled StateGraph ready to invoke.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("ingest_data", node_ingest_data)
    graph.add_node("data_quality", node_data_quality)
    graph.add_node("trend_analysis", node_trend_analysis)
    graph.add_node("anomaly_detection", node_anomaly_detection)
    graph.add_node("nlp_analysis", node_nlp_analysis)
    graph.add_node("opportunity_scoring", node_opportunity_scoring)
    graph.add_node("report_generation", node_report_generation)
    graph.add_node("executive_insights", node_executive_insights)

    # Set entry point
    graph.set_entry_point("ingest_data")

    # Sequential: ingest → quality
    graph.add_edge("ingest_data", "data_quality")

    # Fan-out: quality → [trends, anomalies, nlp] (parallel analysis)
    graph.add_edge("data_quality", "trend_analysis")
    graph.add_edge("data_quality", "anomaly_detection")
    graph.add_edge("data_quality", "nlp_analysis")

    # Fan-in: [trends, anomalies, nlp] → opportunity_scoring
    graph.add_edge("trend_analysis", "opportunity_scoring")
    graph.add_edge("anomaly_detection", "opportunity_scoring")
    graph.add_edge("nlp_analysis", "opportunity_scoring")

    # Sequential: opportunities → report → insights → END
    # (Sequential to avoid Groq rate-limit 429s on free tier)
    graph.add_edge("opportunity_scoring", "report_generation")
    graph.add_edge("report_generation", "executive_insights")
    graph.add_edge("executive_insights", END)

    return graph.compile()


def run_pipeline() -> dict:
    """Execute the full analytical pipeline.

    Returns:
        Final AgentState dict with all results.
    """
    logger.info("Starting analytical pipeline")
    graph = build_graph()

    initial_state: AgentState = {
        "raw_df_path": "",
        "raw_row_count": 0,
        "clean_df_serialized": "",
        "data_quality": {},
        "kpi_summary": {},
        "weekly_trends": {},
        "team_performance": [],
        "channel_performance": [],
        "category_performance": [],
        "week_over_week": {},
        "anomalies": {},
        "chatbot_escalation": {},
        "nlp_summary": {},
        "opportunities": [],
        "weekly_brief": {},
        "report_markdown": "",
        "executive_insights": {},
        "current_step": "initialized",
        "errors": [],
        "execution_log": [],
    }

    result = graph.invoke(initial_state)
    logger.info(
        "Pipeline complete. Steps: %s",
        " → ".join(result.get("execution_log", [])),
    )
    return result
