"""Streamlit application entry point.

AI-Powered Customer Operations Command Center
5-tab dashboard: Dashboard | Opportunities | Trends | NLP | Weekly Brief
"""

import json
import logging

import streamlit as st
import pandas as pd

from src.app.components import (
    render_dashboard_tab,
    render_nlp_tab,
    render_opportunities_tab,
    render_trends_tab,
    render_weekly_brief_tab,
)
from src.app.styles import inject_custom_css
from src.config import GROUPON_GREEN
from src.data_cleaning import clean_data, load_raw_data
from src.analytics import (
    compute_category_performance,
    compute_channel_performance,
    compute_chatbot_escalation_analysis,
    compute_kpi_summary,
    compute_team_performance,
    compute_weekly_trends,
)
from src.nlp_analysis import compute_nlp_summary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ops Command Center | Groupon",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner="Loading ticket data...")
def load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load and clean data with caching."""
    raw = load_raw_data()
    clean, log = clean_data(raw)
    return raw, clean, log


@st.cache_data(show_spinner="Running analytics...")
def run_analytics(clean_json: str) -> dict:
    """Run all non-NLP analytics with caching."""
    df = pd.read_json(clean_json)
    return {
        "kpi": compute_kpi_summary(df),
        "teams": compute_team_performance(df).to_dict("records"),
        "channels": compute_channel_performance(df).to_dict("records"),
        "categories": compute_category_performance(df).to_dict("records"),
        "trends": compute_weekly_trends(
            df,
            [
                "first_response_min",
                "resolution_min",
                "csat_score",
                "cost_usd",
                "contacts_per_ticket",
            ],
        ).to_dict("records"),
        "chatbot": compute_chatbot_escalation_analysis(df),
    }


@st.cache_data(show_spinner="Running NLP analysis...")
def run_nlp(clean_json: str) -> dict:
    """Run NLP analysis with caching."""
    df = pd.read_json(clean_json)
    return compute_nlp_summary(df)


def main() -> None:
    """Main application entry point."""
    inject_custom_css()

    # --- Header ---
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:24px">
            <h1 style="margin:0;color:{GROUPON_GREEN}">📊 Ops Command Center</h1>
            <span style="color:#999;font-size:14px;margin-top:8px">
                AI-Powered Customer Operations Intelligence
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Load data ---
    raw_df, clean_df, cleaning_log = load_and_clean()
    clean_json = clean_df.to_json(orient="records")

    # --- Run analytics ---
    analytics = run_analytics(clean_json)
    nlp_results = run_nlp(clean_json)

    # --- Tabs ---
    tab_dashboard, tab_opps, tab_trends, tab_nlp, tab_brief = st.tabs(
        ["📈 Dashboard", "🎯 Opportunities", "📊 Trends", "💬 NLP Insights", "📋 Weekly Brief"]
    )

    with tab_dashboard:
        render_dashboard_tab(clean_df, analytics)

    with tab_opps:
        render_opportunities_tab(analytics)

    with tab_trends:
        render_trends_tab(clean_df, analytics)

    with tab_nlp:
        render_nlp_tab(clean_df, nlp_results)

    with tab_brief:
        render_weekly_brief_tab(analytics, nlp_results)

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#999;font-size:12px">'
        "Prepared by: AI-Powered Ops Command Center | "
        "Groupon Global Customer Operations</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
