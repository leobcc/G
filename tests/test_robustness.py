"""
Robustness tests — verify the pipeline works on arbitrary uploaded CSV data.

Tests cover:
1. Different week numbers (not hardcoded weeks 7-10)
2. Missing timestamps (NaT values)
3. Edge cases in data cleaning
4. detect_complete_weeks dynamic behavior
5. Analytics functions with column guards
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_cleaning import clean_data, detect_complete_weeks
from analytics import (
    compute_kpi_summary,
    compute_weekly_trends,
    compute_wow_kpis,
    compute_team_performance,
    compute_channel_performance,
    compute_category_performance,
    compute_chatbot_escalation_analysis,
)


def make_synthetic_df(
    n_rows: int = 500,
    start_date: str = "2025-01-06",
    weeks: int = 4,
    include_nat: int = 10,
    include_missing_csat: int = 50,
) -> pd.DataFrame:
    """Generate a synthetic ticket DataFrame with arbitrary dates/weeks."""
    rng = np.random.default_rng(42)
    
    start = pd.Timestamp(start_date)
    # Spread across specified number of weeks
    dates = [start + timedelta(hours=int(rng.integers(0, weeks * 7 * 24))) for _ in range(n_rows)]
    
    # Inject NaT values
    for i in rng.choice(n_rows, size=include_nat, replace=False):
        dates[i] = pd.NaT
    
    channels = rng.choice(["email", "chat", "phone", "social"], size=n_rows)
    categories = rng.choice(["refund", "order_status", "merchant_issue", "billing", "account"], size=n_rows)
    teams = rng.choice(["in_house", "bpo_vendorA", "bpo_vendorB", "ai_chatbot"], size=n_rows)
    statuses = rng.choice(["resolved", "escalated", "abandoned", "pending"], size=n_rows, p=[0.6, 0.15, 0.1, 0.15])
    priorities = rng.choice(["low", "medium", "high", "urgent"], size=n_rows)
    
    csat = rng.uniform(1, 5, size=n_rows).round(1)
    # Set some to NaN
    nan_idx = rng.choice(n_rows, size=include_missing_csat, replace=False)
    csat[nan_idx] = np.nan
    # Set a few out of range
    csat[0] = 0.5
    csat[1] = 6.0
    csat[2] = -1.0
    
    frt = rng.exponential(30, size=n_rows).round(1)
    frt[3] = -5.0  # negative should NOT be cleaned (only resolution_min is cleaned)
    
    res_min = rng.exponential(60, size=n_rows).round(1)
    res_min[4] = -10.0  # negative - should be set to NaN
    res_min[5] = -3.0
    
    cost = rng.uniform(1, 10, size=n_rows).round(2)
    
    df = pd.DataFrame({
        "ticket_id": [f"TKT-{i:05d}" for i in range(n_rows)],
        "created_at": dates,
        "channel": channels,
        "category": categories,
        "subcategory": rng.choice(["sub_a", "sub_b", "sub_c"], size=n_rows),
        "priority": priorities,
        "assigned_team": teams,
        "resolution_status": statuses,
        "first_response_min": frt,
        "resolution_min": res_min,
        "csat_score": csat,
        "customer_message": [f"Test message {i}" for i in range(n_rows)],
        "cost_usd": cost,
        "market": rng.choice(["US", "UK", "DE", "FR"], size=n_rows),
        "customer_contacts": rng.integers(1, 5, size=n_rows),
    })
    
    return df


class TestDetectCompleteWeeks:
    """Test the dynamic week detection function."""

    def test_detects_weeks_from_jan_data(self):
        """Synthetic data starting Jan 6 2025 should detect weeks 2-5 (ISO weeks)."""
        df = make_synthetic_df(n_rows=500, start_date="2025-01-06", weeks=4)
        clean_df, _ = clean_data(df)
        weeks = detect_complete_weeks(clean_df)
        # Should detect at least 2 complete weeks from 4-week span
        assert len(weeks) >= 2
        # Weeks should NOT be 7-10 (those are from the original sample)
        assert not set(weeks).issubset({7, 8, 9, 10})

    def test_single_week_data(self):
        """Single week of data should return that week as complete."""
        df = make_synthetic_df(n_rows=200, start_date="2025-03-10", weeks=1)
        clean_df, _ = clean_data(df)
        weeks = detect_complete_weeks(clean_df)
        assert len(weeks) >= 1

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty list."""
        df = pd.DataFrame(columns=["week_number"])
        weeks = detect_complete_weeks(df)
        assert weeks == []

    def test_all_nat_timestamps(self):
        """If all timestamps are NaT, week_number will be all NA → empty."""
        # Directly test detect_complete_weeks with all-NA week_number
        # (bypassing full clean_data which would fail on KNN with all-NaN features)
        df = pd.DataFrame({
            "week_number": pd.array([pd.NA] * 50, dtype="Int64"),
        })
        weeks = detect_complete_weeks(df)
        assert weeks == []


class TestCleanDataRobustness:
    """Test data cleaning handles edge cases."""

    def test_nat_timestamps_dont_crash(self):
        """NaT timestamps should not crash the cleaning pipeline."""
        df = make_synthetic_df(n_rows=100, include_nat=20)
        clean_df, log = clean_data(df)
        assert len(clean_df) == 100  # No rows dropped
        # week_number should be NA for NaT rows
        nat_count = clean_df["week_number"].isna().sum()
        assert nat_count >= 20  # At least the injected NaTs

    def test_csat_clamping(self):
        """Out-of-range CSAT values should be set to NaN."""
        df = make_synthetic_df(n_rows=100)
        clean_df, log = clean_data(df)
        valid_csat = clean_df["csat_score"].dropna()
        assert (valid_csat >= 1).all()
        assert (valid_csat <= 5).all()
        assert log["csat_clamped"] >= 3  # We set 3 out of range

    def test_negative_resolution_fixed(self):
        """Negative resolution_min values should be NaN."""
        df = make_synthetic_df(n_rows=100)
        clean_df, log = clean_data(df)
        valid_res = clean_df["resolution_min"].dropna()
        assert (valid_res >= 0).all()
        assert log["negative_resolution_fixed"] >= 2


class TestAnalyticsWithDifferentWeeks:
    """Test analytics functions work with non-standard week numbers."""

    @pytest.fixture
    def jan_data(self):
        """Cleaned data from January 2025 (weeks 2-5 approximately)."""
        df = make_synthetic_df(n_rows=500, start_date="2025-01-06", weeks=4)
        clean_df, _ = clean_data(df)
        return clean_df

    def test_kpi_summary_works(self, jan_data):
        """KPI summary should work regardless of week numbers."""
        kpis = compute_kpi_summary(jan_data)
        assert "total_tickets" in kpis
        assert kpis["total_tickets"] == 500

    def test_weekly_trends_dynamic_weeks(self, jan_data):
        """Weekly trends should use detected complete weeks."""
        cw = detect_complete_weeks(jan_data)
        trends = compute_weekly_trends(jan_data, complete_weeks=cw)
        assert len(trends) > 0
        # Verify we're using detected weeks, not hardcoded 7-10
        if "week_number" in trends.columns:
            assert not set(trends["week_number"].tolist()).issubset({7, 8, 9, 10})

    def test_wow_kpis_dynamic_weeks(self, jan_data):
        """WoW KPIs should work with detected weeks."""
        cw = detect_complete_weeks(jan_data)
        if len(cw) >= 2:
            wow = compute_wow_kpis(jan_data, complete_weeks=cw)
            assert "current_week" in wow
            assert wow["current_week"] == cw[-1]

    def test_team_performance_works(self, jan_data):
        """Team performance should compute without crash."""
        result = compute_team_performance(jan_data)
        assert len(result) == 4  # 4 teams in synthetic data
        assert "team" in result.columns

    def test_channel_performance_works(self, jan_data):
        """Channel performance should compute without crash."""
        result = compute_channel_performance(jan_data)
        assert len(result) == 4  # 4 channels
        assert "channel" in result.columns

    def test_category_performance_works(self, jan_data):
        """Category performance should compute without crash."""
        result = compute_category_performance(jan_data)
        assert len(result) == 5  # 5 categories
        assert "category" in result.columns

    def test_chatbot_escalation_works(self, jan_data):
        """Chatbot escalation analysis should not crash."""
        result = compute_chatbot_escalation_analysis(jan_data)
        assert "overall_escalation_rate" in result
        assert isinstance(result["overall_escalation_rate"], float)


class TestMissingColumns:
    """Test that functions handle missing optional columns gracefully."""

    def test_team_perf_missing_cost(self):
        """Team perf should work even without cost_usd column."""
        df = make_synthetic_df(n_rows=100)
        clean_df, _ = clean_data(df)
        clean_df = clean_df.drop(columns=["cost_usd"])
        result = compute_team_performance(clean_df)
        assert len(result) > 0
        assert "cost_usd" not in result.columns or "avg_cost_usd" not in result.columns

    def test_team_perf_missing_groupby_col(self):
        """If groupby column is missing, return empty DataFrame."""
        df = make_synthetic_df(n_rows=100)
        clean_df, _ = clean_data(df)
        clean_df = clean_df.drop(columns=["assigned_team"])
        result = compute_team_performance(clean_df)
        assert len(result) == 0

    def test_channel_perf_missing_csat(self):
        """Channel perf should work without csat_score."""
        df = make_synthetic_df(n_rows=100)
        clean_df, _ = clean_data(df)
        clean_df = clean_df.drop(columns=["csat_score"])
        result = compute_channel_performance(clean_df)
        assert len(result) > 0

    def test_weekly_trends_missing_columns(self):
        """Weekly trends should handle missing metric columns gracefully."""
        df = make_synthetic_df(n_rows=200, start_date="2025-02-03", weeks=3)
        clean_df, _ = clean_data(df)
        clean_df = clean_df.drop(columns=["cost_usd", "csat_score"])
        cw = detect_complete_weeks(clean_df)
        trends = compute_weekly_trends(clean_df, complete_weeks=cw)
        assert len(trends) > 0
        # cost/csat columns should not appear since we dropped them
        assert "avg_cost" not in trends.columns or trends["avg_cost"].isna().all()
