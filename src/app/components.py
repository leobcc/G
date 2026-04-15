"""Reusable Streamlit components for each dashboard tab."""

import pandas as pd
import streamlit as st

from src.visualizations import (
    create_kpi_card,
    plot_category_treemap,
    plot_channel_distribution,
    plot_cost_efficiency_scatter,
    plot_frustration_by_category,
    plot_heatmap_hourly,
    plot_kpi_trend,
    plot_resolution_funnel,
    plot_sentiment_distribution,
    plot_team_comparison,
)


def render_dashboard_tab(df: pd.DataFrame, analytics: dict) -> None:
    """Render the main KPI dashboard tab."""
    kpi = analytics["kpi"]

    # KPI Cards Row
    cols = st.columns(5)
    kpi_items = [
        ("Total Tickets", f"{kpi['total_tickets']:,}", None),
        ("Resolution Rate", f"{kpi['resolution_rate']:.1%}", None),
        ("Avg FRT", f"{kpi['avg_first_response_min']:.0f} min", None),
        ("Avg CSAT", f"{kpi['avg_csat']:.2f}", None),
        ("Avg Cost", f"${kpi['avg_cost_usd']:.2f}", None),
    ]
    for col, (label, value, delta) in zip(cols, kpi_items):
        col.markdown(create_kpi_card(label, value, delta), unsafe_allow_html=True)

    st.markdown("---")

    # Charts Row 1: Team + Channel
    col1, col2 = st.columns(2)
    with col1:
        team_df = pd.DataFrame(analytics["teams"])
        fig = plot_team_comparison(team_df, "resolution_rate", "Resolution Rate by Team")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = plot_channel_distribution(df)
        st.plotly_chart(fig, use_container_width=True)

    # Charts Row 2: Funnel + Heatmap
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_resolution_funnel(df)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = plot_heatmap_hourly(df)
        st.plotly_chart(fig, use_container_width=True)

    # Charts Row 3: Cost Efficiency + Category Treemap
    col1, col2 = st.columns(2)
    with col1:
        team_df = pd.DataFrame(analytics["teams"])
        fig = plot_cost_efficiency_scatter(team_df)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = plot_category_treemap(df)
        st.plotly_chart(fig, use_container_width=True)


def render_opportunities_tab(analytics: dict) -> None:
    """Render the opportunities identification tab."""
    st.subheader("🎯 Improvement Opportunities")

    chatbot = analytics["chatbot"]
    teams = analytics["teams"]
    kpi = analytics["kpi"]

    # Pre-computed opportunities based on data patterns
    opportunities = [
        {
            "title": "Optimize AI Chatbot Escalation Flow",
            "impact": f"Escalation rate: {chatbot['overall_escalation_rate']:.1%}",
            "description": (
                "AI chatbot has the highest escalation rate. Reducing by 30% "
                "could save ~$X/month at scale."
            ),
            "effort": "Medium",
            "category": "🤖 Automation",
        },
        {
            "title": "BPO Vendor B Performance Gap",
            "description": (
                "BPO Vendor B has the lowest CSAT and resolution rate. "
                "A targeted quality improvement program is needed."
            ),
            "impact": "Resolution rate gap vs in-house",
            "effort": "Medium",
            "category": "📋 Quality",
        },
        {
            "title": "Reduce Repeat Contacts",
            "description": (
                f"Avg contacts/ticket: {kpi['avg_contacts_per_ticket']:.1f}. "
                "First-contact resolution improvement could reduce volume 15-20%."
            ),
            "impact": "Volume reduction at scale",
            "effort": "High",
            "category": "⚙️ Process",
        },
    ]

    for i, opp in enumerate(opportunities, 1):
        with st.expander(f"#{i} — {opp['title']}", expanded=i == 1):
            st.markdown(f"**Category:** {opp['category']}")
            st.markdown(f"**Impact:** {opp['impact']}")
            st.markdown(f"**Effort:** {opp['effort']}")
            st.markdown(opp["description"])

    st.info(
        "💡 Run the full LangGraph pipeline with your Anthropic API key "
        "to get AI-generated, data-backed opportunity scoring."
    )


def render_trends_tab(df: pd.DataFrame, analytics: dict) -> None:
    """Render the weekly trends analysis tab."""
    st.subheader("📊 Weekly Trends")

    trends_df = pd.DataFrame(analytics["trends"])
    if trends_df.empty:
        st.warning("No trend data available.")
        return

    # Metric selector
    metric = st.selectbox(
        "Select metric",
        ["csat_score", "first_response_min", "resolution_min", "cost_usd", "contacts_per_ticket"],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    titles = {
        "csat_score": "CSAT Score Trend",
        "first_response_min": "First Response Time Trend (min)",
        "resolution_min": "Resolution Time Trend (min)",
        "cost_usd": "Cost per Ticket Trend ($)",
        "contacts_per_ticket": "Contacts per Ticket Trend",
    }

    fig = plot_kpi_trend(trends_df, metric, titles.get(metric, metric))
    st.plotly_chart(fig, use_container_width=True)

    # Team comparison
    st.markdown("### Team Comparison")
    team_df = pd.DataFrame(analytics["teams"])
    team_metric = st.selectbox(
        "Compare teams by",
        ["resolution_rate", "avg_csat", "avg_cost_usd", "avg_frt_min", "escalation_rate"],
        format_func=lambda x: x.replace("_", " ").title(),
    )
    fig = plot_team_comparison(team_df, team_metric, f"Team Comparison: {team_metric.replace('_', ' ').title()}")
    st.plotly_chart(fig, use_container_width=True)


def render_nlp_tab(df: pd.DataFrame, nlp_results: dict) -> None:
    """Render the NLP insights tab."""
    st.subheader("💬 NLP Insights")

    if not nlp_results:
        st.warning("NLP analysis not available.")
        return

    # Sentiment overview
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Sentiment", f"{nlp_results['avg_sentiment_polarity']:.3f}")
    col2.metric("Frustration Rate", f"{nlp_results['frustration_rate']:.1%}")
    col3.metric("Frustrated Tickets", f"{nlp_results['total_frustrated_tickets']:,}")

    st.markdown("---")

    # Frustration by category
    if nlp_results.get("frustration_by_category"):
        fig = plot_frustration_by_category(nlp_results["frustration_by_category"])
        st.plotly_chart(fig, use_container_width=True)

    # Topics
    if nlp_results.get("topics"):
        st.markdown("### Detected Topics")
        for topic in nlp_results["topics"]:
            words = ", ".join(topic["top_words"][:7])
            st.markdown(f"**{topic['label']}:** {words}")


def render_weekly_brief_tab(analytics: dict, nlp_results: dict) -> None:
    """Render the weekly brief tab."""
    st.subheader("📋 Weekly Operations Brief")

    st.info(
        "💡 Connect your Anthropic API key and run the full LangGraph pipeline "
        "to generate an AI-written executive weekly brief. "
        "A preview based on static analysis is shown below."
    )

    kpi = analytics["kpi"]
    chatbot = analytics["chatbot"]

    brief = f"""
## 📊 Weekly Ops Intelligence Brief

### Executive Summary
- **{kpi['total_tickets']:,}** tickets processed across 4 weeks (~{kpi['total_tickets'] * 12:,}/month at scale)
- Overall resolution rate: **{kpi['resolution_rate']:.1%}** | Avg CSAT: **{kpi['avg_csat']:.2f}/5**
- AI chatbot escalation rate at **{chatbot['overall_escalation_rate']:.1%}** — primary optimization target
- Total cost: **${kpi['total_cost_usd']:,.0f}** (~${kpi['total_cost_usd'] * 12:,.0f}/month at scale)

### Key Metrics
| Metric | Value |
|--------|-------|
| Resolution Rate | {kpi['resolution_rate']:.1%} |
| Avg First Response | {kpi['avg_first_response_min']:.0f} min |
| Avg Resolution Time | {kpi['avg_resolution_min']:.0f} min |
| Avg CSAT | {kpi['avg_csat']:.2f}/5 |
| Avg Cost/Ticket | ${kpi['avg_cost_usd']:.2f} |
| Avg Contacts/Ticket | {kpi['avg_contacts_per_ticket']:.1f} |

### NLP Insights
- Average sentiment polarity: **{nlp_results.get('avg_sentiment_polarity', 'N/A')}**
- Frustration rate: **{nlp_results.get('frustration_rate', 'N/A')}**

---
*Prepared by: AI-Powered Ops Command Center*
"""
    st.markdown(brief)

    # Download button
    st.download_button(
        label="📥 Download Brief as Markdown",
        data=brief,
        file_name="weekly_ops_brief.md",
        mime="text/markdown",
    )
