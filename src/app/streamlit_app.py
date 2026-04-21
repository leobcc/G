"""Streamlit application entry point.

AI-Powered Customer Operations Command Center.

Three-state flow:
  1. Landing  — select data source (sample dataset or CSV upload)
  2. Running  — full-screen progress feedback while pipeline executes
  3. Analysis — sidebar navigation across Dashboard / Opportunities / …
"""

import logging
import sys
import time
from pathlib import Path
from typing import Callable

# Ensure the project root is on sys.path so `from src.*` imports resolve
# regardless of how Streamlit is launched (e.g. `streamlit run src/app/streamlit_app.py`).
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# Streamlit Cloud injects secrets via st.secrets — push to os.environ for libs
import os  # noqa: E402
try:
    for key, val in st.secrets.items():
        if isinstance(val, str) and key not in os.environ:
            os.environ[key] = val
except Exception:
    pass  # No secrets.toml available (local dev uses .env instead)

from src.agent.graph import build_graph  # noqa: E402
from src.analytics import (  # noqa: E402
    compute_category_performance,
    compute_channel_performance,
    compute_chatbot_escalation_analysis,
    compute_kpi_summary,
    compute_team_performance,
    compute_week_date_ranges,
    compute_weekly_trends,
    compute_wow_kpis,
    find_statistical_outliers,
    run_correlation_analysis,
)
from src.app.components import (  # noqa: E402
    render_dashboard_tab,
    render_nlp_tab,
    render_opportunities_tab,
    render_trends_tab,
    render_weekly_brief_tab,
)
from src.app.styles import inject_custom_css  # noqa: E402
from src.config import COMPLETE_WEEKS, OUTPUT_DIR, RAW_DATA_PATH  # noqa: E402
from src.data_cleaning import clean_data, detect_complete_weeks, get_data_quality_report, load_raw_data  # noqa: E402
from src.nlp_analysis import compute_nlp_summary  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "logo.png"
_PAGES = ["Dashboard", "NLP Insights", "Trends", "Opportunities", "Weekly Brief"]
_AGENT_NODE_ORDER = [
    "ingest_data",
    "data_quality",
    "trend_analysis",
    "anomaly_detection",
    "nlp_analysis",
    "opportunity_scoring",
    "report_generation",
    "executive_insights",
]
_AGENT_NODE_LABELS = {
    "ingest_data": "Ingesting ticket data",
    "data_quality": "Cleaning and validating data",
    "trend_analysis": "Computing trend analysis",
    "anomaly_detection": "Detecting anomalies",
    "nlp_analysis": "Running NLP analysis",
    "opportunity_scoring": "Scoring opportunities (agentic)",
    "report_generation": "Generating weekly brief",
    "executive_insights": "Generating executive insights",
}

# Session-state keys managed by this module (used for cleanup)
_STATE_KEYS = [
    "analysis_complete",
    "source_name",
    "clean_df",
    "analytics",
    "nlp_results",
    "agent_result",
    "quality_report",
    "cleaning_log",
    "_running",
    "_data_path",
    "_source_name",
    "_uploaded_raw",
]

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Ops Command Center | Groupon",
    page_icon=str(_LOGO_PATH) if _LOGO_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _compute_analytics(clean_df: pd.DataFrame, complete_weeks: list[int] | None = None) -> dict:
    """Run all statistical analytics on the cleaned DataFrame."""
    # Use dynamically detected weeks, fall back to config default
    if complete_weeks is None:
        complete_weeks = detect_complete_weeks(clean_df)
    if not complete_weeks:
        # Fallback: use COMPLETE_WEEKS from config (never blindly use all weeks)
        complete_weeks = list(COMPLETE_WEEKS)

    # Filter to complete weeks for KPIs
    complete_df = clean_df[clean_df["week_number"].isin(complete_weeks)]
    if complete_df.empty:
        complete_df = clean_df  # fallback: use everything

    kpi = compute_kpi_summary(complete_df)
    derived_trends = compute_weekly_trends(clean_df, complete_weeks=complete_weeks)
    raw_trends = compute_weekly_trends(
        clean_df,
        [
            "first_response_min",
            "resolution_min",
            "csat_score",
            "cost_usd",
            "contacts_per_ticket",
        ],
        complete_weeks=complete_weeks,
    )
    teams = compute_team_performance(complete_df)
    channels = compute_channel_performance(complete_df)
    categories = compute_category_performance(complete_df)
    chatbot = compute_chatbot_escalation_analysis(complete_df)
    anomalies = {}
    for col in ["first_response_min", "resolution_min", "cost_usd", "contacts_per_ticket"]:
        outliers = find_statistical_outliers(clean_df, col, method="iqr", threshold=1.5)
        anomalies[col] = {
            "total_outliers": len(outliers),
            "column": col,
            "method": "iqr",
            "threshold": 1.5,
            "sample": outliers.head(10).to_dict("records"),
        }
    correlations = run_correlation_analysis(
        complete_df,
        target="csat_score",
        features=[
            "first_response_min",
            "resolution_min",
            "cost_usd",
            "contacts_per_ticket",
        ],
    )
    wow = compute_wow_kpis(clean_df, complete_weeks=complete_weeks)
    week_date_ranges = compute_week_date_ranges(clean_df, complete_weeks=complete_weeks)
    return {
        "kpi": kpi,
        "wow": wow,
        "teams": teams.to_dict("records"),
        "channels": channels.to_dict("records"),
        "categories": categories.to_dict("records"),
        "derived_trends": derived_trends.to_dict("records"),
        "raw_trends": raw_trends.to_dict("records"),
        "anomalies": anomalies,
        "chatbot": chatbot,
        "correlations": correlations,
        "week_date_ranges": week_date_ranges,
        "complete_weeks": complete_weeks,
    }


def _run_agent(
    data_path: str,
    precomputed_state: dict | None = None,
    on_event: Callable[[str, dict], None] | None = None,
) -> dict | None:
    """Run the LangGraph agent pipeline and return the final state dict.

    Streams node updates so the caller can show live progress during execution.
    """
    try:
        graph = build_graph()
        initial_state = {
            "raw_df_path": data_path,
            "execution_log": [],
            "errors": [],
        }
        if precomputed_state:
            initial_state.update(precomputed_state)
        final_state = dict(initial_state)

        for update in graph.stream(initial_state, stream_mode="updates"):
            if not isinstance(update, dict):
                continue

            for node_name, delta in update.items():
                if not isinstance(delta, dict):
                    continue

                if on_event is not None:
                    on_event(node_name, delta)

                for key, value in delta.items():
                    if key in ("execution_log", "errors") and isinstance(value, list):
                        final_state.setdefault(key, [])
                        final_state[key].extend(value)
                    else:
                        final_state[key] = value

        return final_state
    except Exception:
        logger.exception("Agent pipeline failed")
        return None


# ── Landing page ───────────────────────────────────────────────────────────


def _render_landing() -> None:
    """Data-source selection screen shown before any analysis runs."""
    with st.sidebar:
        if _LOGO_PATH.exists():
            st.image(str(_LOGO_PATH))
        st.markdown("## Customer Ops Command Center")
        st.caption("Groupon Customer Operations")

    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("")
        st.markdown("### Ops Intelligence Command Center")
        st.caption("AI-powered analytics for customer operations")
        st.divider()
        st.markdown("#### Select a data source to begin")
        st.markdown("")

        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.markdown("**Sample Dataset**")
                st.caption("10,000 tickets  |  4 weeks  |  7 markets  |  4 teams")
                if st.button(
                    "Analyze Sample Data",
                    type="primary",
                ):
                    st.session_state._running = True
                    st.session_state._data_path = str(RAW_DATA_PATH)
                    st.session_state._source_name = "Sample Dataset (10K tickets)"
                    st.rerun()

        with col2:
            with st.container(border=True):
                st.markdown("**Upload Your Own**")
                uploaded = st.file_uploader(
                    "Choose a CSV file",
                    type=["csv"],
                    label_visibility="collapsed",
                )
                if uploaded is not None:
                    if st.button(
                        "Analyze Uploaded Data",
                        type="primary",
                    ):
                        # Persist uploaded bytes so the agent can read from disk
                        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                        save_path = OUTPUT_DIR / "uploaded_data.csv"
                        raw_bytes = uploaded.getvalue()
                        save_path.write_bytes(raw_bytes)
                        # Validate the saved file (read back and check row count)
                        test_df = pd.read_csv(save_path, low_memory=False, nrows=5)
                        if test_df.empty:
                            st.error("Uploaded file appears empty or corrupted.")
                            st.stop()
                        st.session_state._running = True
                        st.session_state._data_path = str(save_path)
                        st.session_state._source_name = uploaded.name
                        st.rerun()


# ── Analysis pipeline (progress screen) ───────────────────────────────────


def _run_analysis() -> None:
    """Execute the full pipeline with real-time progress feedback."""
    with st.sidebar:
        if _LOGO_PATH.exists():
            st.image(str(_LOGO_PATH))
        st.markdown("## Customer Ops Command Center")
        st.caption("Preparing your analysis ...")

    source_name = st.session_state.get("_source_name", "dataset")
    data_path = st.session_state.get("_data_path", str(RAW_DATA_PATH))

    st.header("Analyzing Ticket Data")
    st.caption(f"Source: {source_name}")

    with st.status("Running unified agent pipeline ...", expanded=True) as status:
        precompute_labels = [
            "Precompute: load and clean data",
            "Precompute: compute deterministic analytics",
            "Precompute: run deterministic NLP",
            "Handoff: seed LangGraph with precomputed context",
        ]
        total_steps = len(precompute_labels) + len(_AGENT_NODE_ORDER)
        completed_steps = 0
        graph_nodes_seen: set[str] = set()

        pipeline_progress = st.progress(0, text=f"Agent pipeline step 0/{total_steps}: starting")
        pipeline_activity = st.empty()
        activity_lines: list[str] = []

        def _render_activity() -> None:
            if not activity_lines:
                return
            recent = activity_lines[-10:]
            activity_md = "\n".join(f"- {line}" for line in recent)
            pipeline_activity.markdown(f"**Live agent pipeline activity**\n\n{activity_md}")

        def _record_step(label: str, detail: str | None = None) -> None:
            nonlocal completed_steps
            completed_steps += 1
            pct = int((completed_steps / total_steps) * 100)
            pipeline_progress.progress(
                pct,
                text=f"Agent pipeline step {completed_steps}/{total_steps}: {label}",
            )
            if detail:
                activity_lines.append(detail)
                _render_activity()

        # Step 1 — Data loading & cleaning (shown as agent pipeline stage)
        raw = load_raw_data(data_path)
        clean, log = clean_data(raw)
        quality = get_data_quality_report(raw, clean)
        complete_weeks = detect_complete_weeks(clean)
        _record_step(
            precompute_labels[0],
            f"Loaded {len(raw):,} tickets — fixed {sum(log.values()):,} data issues "
            f"({len(complete_weeks)} complete weeks detected)",
        )

        # Step 2 — Statistical analytics (shown as agent pipeline stage)
        analytics = _compute_analytics(clean, complete_weeks=complete_weeks)
        _record_step(
            precompute_labels[1],
            f"Analyzed {len(analytics['teams'])} teams, "
            f"{len(analytics['channels'])} channels, {len(analytics['categories'])} categories",
        )

        # Step 3 — NLP (shown as agent pipeline stage)
        nlp = compute_nlp_summary(clean)
        frustration = nlp.get("frustration_rate", 0)
        _record_step(
            precompute_labels[2],
            f"Deterministic NLP complete — frustration rate {frustration:.1%}",
        )

        # Step 4 — AI Agent (graph) with precomputed handoff
        precomputed_agent_state = {
            "raw_row_count": len(raw),
            "clean_df_serialized": clean.to_json(orient="records", date_format="iso"),
            "data_quality": quality,
            "kpi_summary": analytics.get("kpi", {}),
            "weekly_trends": analytics.get("derived_trends", []),
            "team_performance": analytics.get("teams", []),
            "channel_performance": analytics.get("channels", []),
            "category_performance": analytics.get("categories", []),
            "wow_kpis": analytics.get("wow", {}),
            "week_date_ranges": analytics.get("week_date_ranges", {}),
            "complete_weeks": analytics.get("complete_weeks", []),
            "anomalies": analytics.get("anomalies", {}),
            "chatbot_escalation": analytics.get("chatbot", {}),
            "nlp_summary": nlp,
        }
        _record_step(
            precompute_labels[3],
            "Precomputed deterministic context passed to LangGraph",
        )

        def _on_agent_event(node_name: str, delta: dict) -> None:
            nonlocal completed_steps
            if node_name not in graph_nodes_seen:
                graph_nodes_seen.add(node_name)
                completed_steps += 1

            pct = int((completed_steps / total_steps) * 100)
            label = _AGENT_NODE_LABELS.get(node_name, node_name.replace("_", " ").title())
            pipeline_progress.progress(
                pct,
                text=f"Agent pipeline step {completed_steps}/{total_steps}: {label}",
            )

            if isinstance(delta.get("execution_log"), list):
                activity_lines.extend(delta["execution_log"])
            if isinstance(delta.get("errors"), list):
                activity_lines.extend([f"ERROR: {err}" for err in delta["errors"]])
            _render_activity()

        agent = _run_agent(
            data_path,
            precomputed_state=precomputed_agent_state,
            on_event=_on_agent_event,
        )
        if agent and agent.get("report_markdown"):
            pipeline_progress.progress(100, text=f"Agent pipeline step {total_steps}/{total_steps}: complete")
            activity_lines.append("AI executive brief generated successfully")
            _render_activity()
        else:
            pipeline_progress.progress(100, text=f"Agent pipeline step {total_steps}/{total_steps}: fallback mode")
            activity_lines.append("AI brief unavailable — using deterministic fallback")
            _render_activity()

        status.update(label="Analysis complete!", state="complete", expanded=False)

    # Persist results
    st.session_state.update(
        {
            "analysis_complete": True,
            "source_name": source_name,
            "clean_df": clean,
            "analytics": analytics,
            "nlp_results": nlp,
            "agent_result": agent,
            "quality_report": quality,
            "cleaning_log": log,
        }
    )

    # Cleanup temporary keys
    for key in ("_running", "_data_path", "_source_name", "_uploaded_raw"):
        st.session_state.pop(key, None)

    time.sleep(0.5)
    st.rerun()


# ── Analysis pages ─────────────────────────────────────────────────────────


def _render_app() -> None:
    """Render the sidebar navigation and the selected analysis page."""
    clean_df: pd.DataFrame = st.session_state.clean_df
    analytics: dict = st.session_state.analytics
    nlp_results: dict = st.session_state.nlp_results
    quality_report: dict = st.session_state.quality_report
    cleaning_log: dict = st.session_state.cleaning_log

    # ── Sidebar ──
    with st.sidebar:
        if _LOGO_PATH.exists():
            st.image(str(_LOGO_PATH))
        st.markdown("## Customer Ops Command Center")

        st.divider()

        page = st.radio("Navigation", _PAGES, label_visibility="collapsed")

        st.divider()

        st.caption(f"Source: {st.session_state.get('source_name', 'dataset')}")
        if st.button("New Analysis"):
            for key in _STATE_KEYS:
                st.session_state.pop(key, None)
            st.rerun()

    # ── Scroll to top on page change ──
    _prev_page = st.session_state.get("_prev_page", "")
    _page_changed = page != _prev_page
    if _page_changed:
        st.session_state["_prev_page"] = page
        st.session_state["_scroll_key"] = st.session_state.get("_scroll_key", 0) + 1

    # ── Show overlay BEFORE page content renders ──
    # This ensures the overlay is visible immediately, hiding any content flash.
    if _page_changed:
        _sk = st.session_state.get("_scroll_key", 0)
        components.html(
            f"""
            <div style="display:none">{_sk}</div>
            <script>
            (function() {{
                var doc = window.parent.document;
                var main = doc.querySelector('[data-testid="stMain"]')
                         || doc.querySelector('section.main');
                if (!main) return;

                // Scroll to top immediately
                main.scrollTop = 0;
                var bc = main.querySelector('[data-testid="stMainBlockContainer"]')
                       || main.querySelector('.block-container');
                if (bc) bc.scrollTop = 0;

                // Add spinner keyframes if not already present
                if (!doc.getElementById('__spin_style')) {{
                    var style = doc.createElement('style');
                    style.id = '__spin_style';
                    style.textContent = '@keyframes __spin{{to{{transform:rotate(360deg)}}}}';
                    doc.head.appendChild(style);
                }}

                // Ensure main has relative positioning for the overlay
                var mainPos = window.parent.getComputedStyle(main).position;
                if (mainPos === 'static') main.style.position = 'relative';

                // Remove any existing overlay first
                var old = doc.getElementById('__page_loading_overlay');
                if (old) old.remove();

                // Create loading overlay ONLY over main content area
                var overlay = doc.createElement('div');
                overlay.id = '__page_loading_overlay';
                overlay.style.cssText = 'position:absolute;top:0;left:0;width:100%;'
                    + 'height:100%;background:rgba(255,255,255,0.98);z-index:99999;'
                    + 'display:flex;align-items:center;justify-content:center;'
                    + 'flex-direction:column;gap:0.75rem;';
                overlay.innerHTML = '<div style="width:36px;height:36px;'
                    + 'border:3px solid #e0e0e0;border-top:3px solid #53A318;'
                    + 'border-radius:50%;animation:__spin 0.7s linear infinite"></div>'
                    + '<p style="color:#7a7868;font-size:0.9rem;margin:0">Loading…</p>';
                main.appendChild(overlay);

                // Scroll again after a tick
                setTimeout(function() {{
                    main.scrollTop = 0;
                    if (bc) bc.scrollTop = 0;
                }}, 30);
            }})();
            </script>
            """,
            height=0,
        )

    # ── Page routing ──
    if page == "Dashboard":
        st.header("Dashboard")
        render_dashboard_tab(clean_df, analytics)
    elif page == "Opportunities":
        st.header("Improvement Opportunities")
        render_opportunities_tab(clean_df, analytics, nlp_results)
    elif page == "Trends":
        st.header("Trends & Comparisons")
        render_trends_tab(clean_df, analytics)
    elif page == "NLP Insights":
        st.header("NLP Insights")
        render_nlp_tab(clean_df, nlp_results)
    elif page == "Weekly Brief":
        st.header("Weekly Operations Brief")
        render_weekly_brief_tab(analytics, nlp_results, quality_report, cleaning_log)

    # ── Remove overlay AFTER page content has rendered ──
    if _page_changed:
        components.html(
            """
            <script>
            (function() {
                var doc = window.parent.document;
                // Give charts/content a moment to fully render, then fade out
                setTimeout(function() {
                    var overlay = doc.getElementById('__page_loading_overlay');
                    if (overlay) {
                        overlay.style.transition = 'opacity 0.2s';
                        overlay.style.opacity = '0';
                        setTimeout(function() { overlay.remove(); }, 220);
                    }
                }, 400);
            })();
            </script>
            """,
            height=0,
        )

    # ── Footer ──
    st.divider()
    st.caption(
        "Prepared by: AI-Powered Customer Ops Command Center  |  "
        "Groupon Global Customer Operations"
    )


# ── Entry point ────────────────────────────────────────────────────────────


def main() -> None:
    """Application entry point — routes to the appropriate screen."""
    inject_custom_css()

    if st.session_state.get("analysis_complete"):
        _render_app()
    elif st.session_state.get("_running"):
        _run_analysis()
    else:
        _render_landing()


if __name__ == "__main__":
    main()
