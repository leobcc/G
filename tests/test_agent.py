"""Tests for the LangGraph agent pipeline."""

import pytest


class TestGraphCompilation:
    """Test that the graph compiles and has expected structure."""

    def test_graph_compiles(self):
        """Graph should compile without errors."""
        from src.agent.graph import build_graph

        graph = build_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self):
        """Graph should contain all expected nodes."""
        from src.agent.graph import build_graph

        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        expected = {
            "ingest_data",
            "data_quality",
            "trend_analysis",
            "anomaly_detection",
            "nlp_analysis",
            "opportunity_scoring",
            "report_generation",
            "executive_insights",
            "__start__",
            "__end__",
        }
        assert expected.issubset(node_names)


class TestAgentState:
    """Test state definitions."""

    def test_state_has_required_keys(self):
        """AgentState should have all required keys."""
        from src.agent.state import AgentState

        annotations = AgentState.__annotations__
        required_keys = [
            "raw_df_path",
            "clean_df_serialized",
            "kpi_summary",
            "opportunities",
            "report_markdown",
            "executive_insights",
            "execution_log",
        ]
        for key in required_keys:
            assert key in annotations, f"Missing key: {key}"


class TestTools:
    """Test tool functions with real data."""

    @pytest.fixture(scope="class")
    def clean_df(self):
        from src.data_cleaning import load_raw_data, clean_data

        raw = load_raw_data()
        df, _ = clean_data(raw)
        return df

    def test_tool_compute_kpis(self, clean_df):
        import json
        from src.agent.tools import tool_compute_kpis

        result = json.loads(tool_compute_kpis(clean_df))
        assert "total_tickets" in result
        assert result["total_tickets"] == 10_000

    def test_tool_team_performance(self, clean_df):
        import json
        from src.agent.tools import tool_team_performance

        result = json.loads(tool_team_performance(clean_df))
        assert isinstance(result, list)
        assert len(result) == 4

    def test_tool_nlp_analysis(self, clean_df):
        import json
        from src.agent.tools import tool_nlp_analysis

        result = json.loads(tool_nlp_analysis(clean_df))
        assert "avg_sentiment_polarity" in result
        assert "frustration_rate" in result
