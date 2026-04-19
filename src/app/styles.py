"""Custom CSS styles for the Streamlit application.

Warm, professional theme inspired by Anthropic's design language,
adapted with Groupon brand green (#53A318) as the primary accent.
"""

import streamlit as st

# ── Theme palette (matches .streamlit/config.toml) ────────────────────────
ACCENT = "#53A318"
ACCENT_HOVER = "#468f15"
TEXT_DARK = "#3d3a2a"
BORDER = "#d3d2ca"


def inject_custom_css() -> None:
    """Inject custom CSS that complements the config.toml theme."""
    st.markdown(
        f"""
        <style>
        /* ── Hide default Streamlit footer ─────────────────────────── */
        footer {{visibility: hidden;}}

        /* ── Metric cards ──────────────────────────────────────────── */
        [data-testid="stMetric"] {{
            border-left: 3px solid {ACCENT};
            border-radius: 0.75rem;
            padding: 14px 16px;
        }}
        [data-testid="stMetricValue"] {{
            font-size: 1.6rem;
            font-weight: 600;
            color: {TEXT_DARK};
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #7a7868;
        }}

        /* ── Tab strip ─────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            border-bottom: 1px solid {BORDER};
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 44px;
            border-radius: 0.6rem 0.6rem 0 0;
            padding: 0 18px;
            font-weight: 500;
            font-size: 0.9rem;
        }}
        .stTabs [aria-selected="true"] {{
            border-bottom: 3px solid {ACCENT} !important;
        }}

        /* ── Remove empty space above page content ─────────────────── */
        .block-container {{
            padding-top: 3rem !important;
            padding-bottom: 2rem !important;
        }}
        header[data-testid="stHeader"] {{
            background: transparent !important;
        }}

        /* ── Ensure sidebar toggle works by keeping the header interactable but hiding deploy/menu ─────────────────── */
        .stAppDeployButton {{
            display: none !important;
        }}
        .stAppDeployButton + div {{
            /* This is typically the main menu (three dots) */
            display: none !important;
        }}

        /* ── Sidebar navigation radio ──────────────────────────────── */
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child,
        [data-testid="stSidebar"] [data-testid="stRadio"] label > div:not(:last-child) {{
            display: none !important;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label {{
            padding: 8px 0px;
            margin-bottom: 4px;
            cursor: pointer;
            background-color: transparent !important;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label p {{
            font-size: 1.05rem;
            font-weight: 500;
            color: #7a7868;
            padding-bottom: 4px;
            border-bottom: 2px solid transparent;
            display: inline-block;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
        [data-testid="stSidebar"] [data-testid="stRadio"] label div[role="radio"][aria-checked="true"] + div p,
        [data-testid="stSidebar"] [data-testid="stRadio"] label div[role="radio"][tabindex="0"] + div p,
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] p {{
            color: {ACCENT} !important;
            font-weight: 600 !important;
            border-bottom: 2px solid {ACCENT} !important;
        }}

        /* ── Download / primary buttons ────────────────────────────── */
        .stDownloadButton > button {{
            background-color: {ACCENT};
            color: white;
            border: none;
            border-radius: 9999px;
            padding: 8px 24px;
            font-weight: 500;
        }}
        .stDownloadButton > button:hover {{
            background-color: {ACCENT_HOVER};
        }}

        /* ── Expander headers ──────────────────────────────────────── */
        .streamlit-expanderHeader {{
            font-weight: 500;
            font-size: 0.95rem;
        }}

        /* ── Alert / info boxes ────────────────────────────────────── */
        .stAlert {{
            border-radius: 0.75rem;
        }}

        /* ── Plotly chart containers ───────────────────────────────── */
        .stPlotlyChart {{
            border-radius: 0.75rem;
        }}

        /* ── Page transition fade-in ───────────────────────────────── */
        @keyframes pageFadeIn {{
            from {{ opacity: 0; }}
            to   {{ opacity: 1; }}
        }}
        [data-testid="stMainBlockContainer"] {{
            animation: pageFadeIn 0.25s ease-in;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
