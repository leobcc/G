"""LangGraph state definitions for the analytical pipeline.

The AgentState TypedDict flows through every node in the graph.
Each node reads what it needs and writes its output key.
"""

from typing import Any, TypedDict


class DataQualityReport(TypedDict):
    """Summary of data quality checks."""
    total_rows: int
    total_columns: int
    missing_values: dict[str, dict]
    cleaning_log: dict[str, int]
    completeness_score: float
    date_range: dict[str, str]


class OpportunityItem(TypedDict):
    """A single improvement opportunity identified by the agent."""
    id: str
    title: str
    description: str
    impact_estimate: str
    effort: str  # "low", "medium", "high"
    priority_score: float  # 0-100
    category: str  # "cost", "quality", "process", "automation"
    supporting_data: dict[str, Any]
    recommendation: str


class WeeklyBrief(TypedDict):
    """Weekly executive brief content."""
    week_number: int
    headline: str
    kpi_summary: dict[str, Any]
    top_insights: list[str]
    opportunities: list[OpportunityItem]
    risk_alerts: list[str]
    recommendations: list[str]
    comparison_vs_prior: dict[str, Any]


class AgentState(TypedDict):
    """Full state object passed through the LangGraph pipeline.

    Each node reads its required inputs and writes to its output key(s).
    """
    # --- Phase 1: Data Ingestion ---
    raw_df_path: str
    raw_row_count: int

    # --- Phase 2: Data Quality ---
    clean_df_serialized: str  # JSON-serialized cleaned DataFrame
    data_quality: DataQualityReport

    # --- Phase 3a: Trend Analysis ---
    kpi_summary: dict[str, Any]
    weekly_trends: dict[str, Any]
    team_performance: list[dict]
    channel_performance: list[dict]
    category_performance: list[dict]
    week_over_week: dict[str, dict]

    # --- Phase 3b: Anomaly Detection ---
    anomalies: dict[str, Any]
    chatbot_escalation: dict[str, Any]

    # --- Phase 3c: NLP Analysis ---
    nlp_summary: dict[str, Any]

    # --- Phase 4: Opportunity Scoring ---
    opportunities: list[OpportunityItem]

    # --- Phase 5: Report Generation ---
    weekly_brief: WeeklyBrief
    report_markdown: str

    # --- Metadata ---
    current_step: str
    errors: list[str]
    execution_log: list[str]
