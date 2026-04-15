"""Tests for the data cleaning module."""

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import clean_data, load_raw_data


class TestLoadRawData:
    """Tests for load_raw_data."""

    def test_loads_csv(self):
        """Verify CSV loads with expected shape."""
        df = load_raw_data()
        assert len(df) == 10_000
        assert "ticket_id" in df.columns

    def test_has_expected_columns(self):
        """Verify all expected columns are present."""
        df = load_raw_data()
        expected = {
            "ticket_id", "created_at", "market", "channel", "category",
            "subcategory", "priority", "assigned_team", "first_response_min",
            "resolution_min", "resolution_status", "csat_score",
            "contacts_per_ticket", "cost_usd", "customer_text",
        }
        assert expected.issubset(set(df.columns))


class TestCleanData:
    """Tests for clean_data."""

    @pytest.fixture
    def raw_df(self):
        return load_raw_data()

    def test_normalizes_markets(self, raw_df):
        """Dirty market labels should be normalized."""
        clean_df, log = clean_data(raw_df)
        assert "United Kingdom" not in clean_df["market"].values
        assert "GER" not in clean_df["market"].values
        assert "USA" not in clean_df["market"].values

    def test_clamps_csat(self, raw_df):
        """CSAT values outside 1-5 should become NaN."""
        clean_df, _ = clean_data(raw_df)
        valid_csat = clean_df["csat_score"].dropna()
        assert valid_csat.min() >= 1
        assert valid_csat.max() <= 5

    def test_fixes_negative_resolution(self, raw_df):
        """Negative resolution_min should become NaN."""
        clean_df, _ = clean_data(raw_df)
        valid_res = clean_df["resolution_min"].dropna()
        assert valid_res.min() >= 0

    def test_adds_derived_columns(self, raw_df):
        """Derived columns should be added."""
        clean_df, _ = clean_data(raw_df)
        assert "week_number" in clean_df.columns
        assert "day_of_week" in clean_df.columns
        assert "is_resolved" in clean_df.columns

    def test_preserves_row_count(self, raw_df):
        """Row count should not change."""
        clean_df, _ = clean_data(raw_df)
        assert len(clean_df) == len(raw_df)
