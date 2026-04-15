"""Custom CSS styles for the Streamlit application."""

import streamlit as st

from src.config import GROUPON_GREEN, GROUPON_DARK


def inject_custom_css() -> None:
    """Inject custom CSS to brand the Streamlit app."""
    st.markdown(
        f"""
        <style>
        /* --- Global --- */
        .stApp {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        /* --- Tab styling --- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 48px;
            border-radius: 8px 8px 0 0;
            padding: 0 20px;
            font-weight: 600;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {GROUPON_GREEN} !important;
            color: white !important;
        }}

        /* --- Metric cards --- */
        [data-testid="stMetric"] {{
            background: linear-gradient(135deg, #f8f9fa, #ffffff);
            border-left: 4px solid {GROUPON_GREEN};
            border-radius: 8px;
            padding: 16px;
        }}
        [data-testid="stMetricValue"] {{
            font-size: 24px;
            font-weight: 700;
            color: {GROUPON_DARK};
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 12px;
            text-transform: uppercase;
            color: #666;
        }}

        /* --- Expander --- */
        .streamlit-expanderHeader {{
            font-weight: 600;
            font-size: 15px;
        }}

        /* --- Download button --- */
        .stDownloadButton > button {{
            background-color: {GROUPON_GREEN};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 24px;
            font-weight: 600;
        }}
        .stDownloadButton > button:hover {{
            background-color: #468f15;
        }}

        /* --- Sidebar --- */
        [data-testid="stSidebar"] {{
            background-color: #f8f9fa;
        }}

        /* --- Info boxes --- */
        .stAlert {{
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
