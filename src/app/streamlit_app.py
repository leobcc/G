"""Streamlit application entry point.

AI-Powered Customer Operations Command Center
5-tab dashboard: Dashboard | Opportunities | Trends | NLP | Weekly Brief
"""

import logging

import pandas as pd
import streamlit as st

from src.app.components import (
    render_dashboard_tab,
    render_nlp_tab,
    render_opportunities_tab,
    render_trends_tab,
    render_weekly_brief_tab,
)
from src.app.styles import inject_custom_css
from src.config import COMPLETE_WEEKS, GROUPON_GREEN, SCALE_FACTOR
from src.data_cleaning import clean_data, get_data_quality_report, load_raw_data
from src.analytics import (
    compute_category_performance,
    compute_channel_performance,
    compute_chatbot_escalation_analysis,
    compute_kpi_summary,
    compute_team_performance,
    compute_weekly_trends,
    run_correlation_analysis,
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


@st.cache_data(show_spinner="Loading ticket data …")
def load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Load and clean data with caching."""
    raw = load_raw_data()
    clean, log = clean_data(raw)
    quality = get_data_quality_report(raw, clean)
    return raw, clean, log, quality


@st.cache_data(show_spinner="Running analytics …")
def run_analytics(_clean_df: pd.DataFrame) -> dict:
    """Run all non-NLP analytics with caching.

    Accepts a DataFrame directly (underscore prefix tells Streamlit
    to skip hashing this unhashable arg).
    """
    df = _clean_df
    kpi = compute_kpi_summary(df)

    # Derived-metric trends (resolution_rate, avg_csat, etc.)
    derived_trends = compute_weekly_trends(df)  # default = derived KPIs
    # Raw-column trends for deep-dive selectors
    raw_trends = compute_weekly_trends(
        df,
        [
            "first_response_min",
            "resolution_min",
            "csat_score",
            "cost_usd",
            "contacts_per_ticket",
        ],
    )

    teams = compute_team_performance(df)
    channels = compute_channel_performance(df)
    categories = compute_category_performance(df)
    chatbot = compute_chatbot_escalation_analysis(df)

    correlations = run_correlation_analysis(
        df,
        target="csat_score",
        features=[
            "first_response_min",
            "resolution_min",
            "cost_usd",
            "contacts_per_ticket",
        ],
    )

    return {
        "kpi": kpi,
        "teams": teams.to_dict("records"),
        "channels": channels.to_dict("records"),
        "categories": categories.to_dict("records"),
        "derived_trends": derived_trends.to_dict("records"),
        "raw_trends": raw_trends.to_dict("records"),
        "chatbot": chatbot,
        "correlations": correlations,
    }


@st.cache_data(show_spinner="Running NLP analysis …")
def run_nlp(_clean_df: pd.DataFrame) -> dict:
    """Run NLP analysis with caching."""
    return compute_nlp_summary(_clean_df)


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
    raw_df, clean_df, cleaning_log, quality_report = load_and_clean()

    # --- Run analytics ---
    analytics = run_analytics(clean_df)
    nlp_results = run_nlp(clean_df)

    # --- Sidebar summary ---
    with st.sidebar:
        st.markdown("### 📋 Data Summary")
        st.metric("Rows loaded", f"{len(clean_df):,}")
        st.metric("Date range", f"Weeks {min(COMPLETE_WEEKS)}–{max(COMPLETE_WEEKS)}")
        st.metric("Scale factor", f"{SCALE_FACTOR}×")

        with st.expander("🔧 Data Quality"):
            st.json(
                {
                    "rows": quality_report["total_rows"],
                    "completeness": f"{quality_report['completeness_score']:.1%}",
                    "fixes_applied": cleaning_log,
                }
            )

    # --- Tabs ---
    tab_dashboard, tab_opps, tab_trends, tab_nlp, tab_brief = st.tabs(
        [
            "📈 Dashboard",
            "🎯 Opportunities",
            "📊 Trends",
            "💬 NLP Insights",
            "📋 Weekly Brief",
        ]
    )

    with tab_dashboard:
        render_dashboard_tab(clean_df, analytics)

    with tab_opps:
        render_opportunities_tab(clean_df, analytics, nlp_results)

    with tab_trends:
        render_trends_tab(clean_df, analytics)

    with tab_nlp:
        render_nlp_tab(clean_df, nlp_results)

    with tab_brief:
        render_weekly_brief_tab(analytics, nlp_results, quality_report, cleaning_log)

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#999;font-size:12px">'
        "Prepared by: AI-Powered Ops Command Center · "
        "Groupon Global Customer Operations</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
