"""Tests for the analytics module."""

import pandas as pd
import pytest

from src.analytics import (
    compute_kpi_summary,
    compute_team_performance,
    compute_channel_performance,
    compute_chatbot_escalation_analysis,
)
from src.data_cleaning import clean_data, load_raw_data


@pytest.fixture(scope="module")
def clean_df():
    """Load and clean data once for all tests in this module."""
    raw = load_raw_data()
    clean, _ = clean_data(raw)
    return clean


class TestKPISummary:
    def test_returns_all_kpis(self, clean_df):
        kpi = compute_kpi_summary(clean_df)
        assert "total_tickets" in kpi
        assert "resolution_rate" in kpi
        assert "avg_csat" in kpi
        assert "avg_cost_usd" in kpi

    def test_resolution_rate_in_range(self, clean_df):
        kpi = compute_kpi_summary(clean_df)
        assert 0 <= kpi["resolution_rate"] <= 1

    def test_filter_by_week(self, clean_df):
        kpi = compute_kpi_summary(clean_df, week=8)
        assert kpi["total_tickets"] < len(clean_df)


class TestTeamPerformance:
    def test_returns_all_teams(self, clean_df):
        teams = compute_team_performance(clean_df)
        assert len(teams) == 4  # in_house, bpo_vendorA, bpo_vendorB, ai_chatbot

    def test_has_required_columns(self, clean_df):
        teams = compute_team_performance(clean_df)
        required = {"team", "resolution_rate", "avg_cost_usd", "avg_csat"}
        assert required.issubset(set(teams.columns))


class TestChatbotEscalation:
    def test_returns_escalation_data(self, clean_df):
        result = compute_chatbot_escalation_analysis(clean_df)
        assert "overall_escalation_rate" in result
        assert "by_category" in result
        assert result["overall_escalation_rate"] > 0
