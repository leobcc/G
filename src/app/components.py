"""Reusable Streamlit components for each dashboard tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import SCALE_FACTOR, COMPLETE_WEEKS
from src.nlp_analysis import add_sentiment_columns
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


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Dashboard Overview
# ──────────────────────────────────────────────────────────────────────────────
def render_dashboard_tab(df: pd.DataFrame, analytics: dict) -> None:
    """Render the main KPI dashboard tab."""
    kpi = analytics["kpi"]

    # KPI Cards Row
    cols = st.columns(5)
    kpi_items = [
        ("Total Tickets", f"{kpi['total_tickets']:,}", None),
        ("Resolution Rate", f"{kpi['resolution_rate']:.1%}", None),
        ("Avg FRT", f"{kpi['avg_first_response_min']:.0f} min", None),
        ("Avg CSAT", f"{kpi['avg_csat']:.2f}/5", None),
        ("Avg Cost", f"${kpi['avg_cost_usd']:.2f}", None),
    ]
    for col, (label, value, delta) in zip(cols, kpi_items):
        col.markdown(create_kpi_card(label, value, delta), unsafe_allow_html=True)

    st.markdown("")  # spacer

    # Secondary KPI row
    cols2 = st.columns(4)
    cols2[0].metric("Escalation Rate", f"{kpi['escalation_rate']:.1%}")
    cols2[1].metric("Abandonment Rate", f"{kpi['abandonment_rate']:.1%}")
    cols2[2].metric("Avg Contacts/Ticket", f"{kpi['avg_contacts_per_ticket']:.1f}")
    cols2[3].metric(
        "Total Cost",
        f"${kpi['total_cost_usd']:,.0f}",
        f"~${kpi['total_cost_usd'] * SCALE_FACTOR:,.0f}/mo at scale",
    )

    st.markdown("---")

    # Charts Row 1: Team + Channel
    col1, col2 = st.columns(2)
    with col1:
        team_df = pd.DataFrame(analytics["teams"])
        fig = plot_team_comparison(
            team_df, "resolution_rate", "Resolution Rate by Team"
        )
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = plot_channel_distribution(df)
        st.plotly_chart(fig, width="stretch")

    # Charts Row 2: Funnel + Heatmap
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_resolution_funnel(df)
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = plot_heatmap_hourly(df)
        st.plotly_chart(fig, width="stretch")

    # Charts Row 3: Cost Efficiency + Category Treemap
    col1, col2 = st.columns(2)
    with col1:
        team_df = pd.DataFrame(analytics["teams"])
        fig = plot_cost_efficiency_scatter(team_df)
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = plot_category_treemap(df)
        st.plotly_chart(fig, width="stretch")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — Opportunities
# ──────────────────────────────────────────────────────────────────────────────
def _build_opportunities(
    analytics: dict, nlp_results: dict, df: pd.DataFrame
) -> list[dict]:
    """Build data-backed improvement opportunities from analytics."""
    kpi = analytics["kpi"]
    chatbot = analytics["chatbot"]
    teams_list = analytics["teams"]
    channels_list = analytics["channels"]

    # Find BPO Vendor B stats
    bpo_b = next(
        (t for t in teams_list if t.get("team") == "bpo_vendorB"), {}
    )
    in_house = next(
        (t for t in teams_list if t.get("team") == "in_house"), {}
    )
    email_ch = next(
        (c for c in channels_list if c.get("channel") == "email"), {}
    )

    # Chatbot escalation savings
    chatbot_esc_rate = chatbot.get("overall_escalation_rate", 0.257)
    chatbot_tickets_month = kpi["total_tickets"] * SCALE_FACTOR * 0.25  # ~25% chatbot
    escalated_monthly = chatbot_tickets_month * chatbot_esc_rate
    cost_per_escalation = 3.50  # avg human handling cost
    chatbot_savings_annual = escalated_monthly * cost_per_escalation * 0.30 * 12  # 30% reduction

    # BPO VB cost of poor quality
    bpo_b_csat = bpo_b.get("avg_csat", 3.02)
    bpo_b_vol_pct = bpo_b.get("ticket_count", 2500) / kpi["total_tickets"]
    bpo_b_monthly_tickets = kpi["total_tickets"] * SCALE_FACTOR * bpo_b_vol_pct
    csat_gap = (in_house.get("avg_csat", 3.94) - bpo_b_csat)
    bpo_b_impact_annual = bpo_b_monthly_tickets * csat_gap * 2.0 * 12  # $2 per CSAT point

    # Email deflection
    email_vol = email_ch.get("ticket_count", 3490)
    email_pct = email_vol / kpi["total_tickets"]
    email_avg_cost = email_ch.get("avg_cost_usd", 3.67)
    chat_avg_cost = 1.50  # estimated chat cost
    deflectable = email_vol * 0.20  # 20% deflectable
    email_savings_annual = deflectable * (email_avg_cost - chat_avg_cost) * SCALE_FACTOR * 12

    # Abandonment recovery
    abandon_rate = kpi["abandonment_rate"]
    abandoned_monthly = kpi["total_tickets"] * SCALE_FACTOR * abandon_rate
    ltv_per_customer = 45  # estimated Groupon LTV
    recovery_rate = 0.25
    abandon_impact_annual = abandoned_monthly * recovery_rate * ltv_per_customer * 12

    # FCR improvement
    avg_contacts = kpi["avg_contacts_per_ticket"]
    multi_contact_pct = (
        len(df[df["contacts_per_ticket"] > 1]) / len(df) if len(df) > 0 else 0.5
    )
    fcr_savings_annual = (
        kpi["total_tickets"]
        * SCALE_FACTOR
        * multi_contact_pct
        * 0.15
        * kpi["avg_cost_usd"]
        * 12
    )

    opportunities = [
        {
            "rank": 1,
            "title": "Optimize AI Chatbot Escalation Flow",
            "impact_usd": chatbot_savings_annual,
            "current": f"Escalation rate: {chatbot_esc_rate:.1%}",
            "target": f"Target: {chatbot_esc_rate * 0.70:.1%} (-30%)",
            "root_cause": (
                f"Chatbot has the highest escalation rate ({chatbot_esc_rate:.1%}) "
                f"across all teams. Categories like merchant_issue and billing "
                f"are being routed to the chatbot but require human judgment."
            ),
            "solution": (
                "1) Implement intent-based routing to bypass chatbot for complex categories\n"
                "2) Fine-tune chatbot on resolved ticket patterns\n"
                "3) Add escalation-prediction model to pre-route high-risk tickets"
            ),
            "effort": "Medium",
            "timeline": "6-8 weeks",
            "owner": "AI/ML Team + Ops Lead",
            "priority": "P0",
            "category": "🤖 Automation",
        },
        {
            "rank": 2,
            "title": "BPO Vendor B Quality Improvement Program",
            "impact_usd": bpo_b_impact_annual,
            "current": f"CSAT: {bpo_b_csat:.2f} | Res rate: {bpo_b.get('resolution_rate', 0):.1%}",
            "target": f"CSAT: {bpo_b_csat + csat_gap * 0.5:.2f} | Close gap by 50%",
            "root_cause": (
                f"BPO Vendor B has the lowest CSAT ({bpo_b_csat:.2f} vs {in_house.get('avg_csat', 3.94):.2f} in-house), "
                f"slowest FRT ({bpo_b.get('avg_frt_min', 80):.0f} min), and highest "
                f"resolution time. Systematic training gap vs other teams."
            ),
            "solution": (
                "1) Implement real-time QA scoring with AI sentiment analysis\n"
                "2) Weekly coaching sessions with top-performer playbooks\n"
                "3) Consider SLA-based contract renegotiation with penalty clauses"
            ),
            "effort": "Medium",
            "timeline": "4-6 weeks",
            "owner": "BPO Program Manager",
            "priority": "P0",
            "category": "📋 Quality",
        },
        {
            "rank": 3,
            "title": "Email-to-Chat Channel Deflection",
            "impact_usd": email_savings_annual,
            "current": f"Email: {email_pct:.0%} of volume, ${email_avg_cost:.2f}/ticket",
            "target": "Deflect 20% of email to chat → save ${:.0f}/year".format(
                email_savings_annual
            ),
            "root_cause": (
                f"Email handles {email_pct:.0%} of tickets at ${email_avg_cost:.2f}/ticket "
                f"with {email_ch.get('avg_frt_min', 79):.0f} min avg FRT. "
                f"Many email queries (order status, voucher issues) are suitable for chat."
            ),
            "solution": (
                "1) Add smart chat widget with proactive engagement on high-deflection pages\n"
                "2) Implement email auto-responder that offers live chat option\n"
                "3) A/B test chat-first vs email-first for common categories"
            ),
            "effort": "Low",
            "timeline": "3-4 weeks",
            "owner": "Digital Experience Team",
            "priority": "P1",
            "category": "💰 Cost",
        },
        {
            "rank": 4,
            "title": "Reduce Customer Abandonment",
            "impact_usd": abandon_impact_annual,
            "current": f"Abandonment rate: {abandon_rate:.1%} (~{abandoned_monthly:,.0f}/mo at scale)",
            "target": f"Reduce to {abandon_rate * 0.75:.1%} with proactive outreach",
            "root_cause": (
                f"{abandon_rate:.1%} of tickets are abandoned — customers who gave up. "
                f"Correlates strongly with long FRT. "
                f"Each abandoned customer represents ~${ltv_per_customer} in potential LTV lost."
            ),
            "solution": (
                "1) Implement SLA-based alerts for tickets approaching abandonment threshold\n"
                "2) Auto-trigger 'We're still here' nudges at 30-min mark\n"
                "3) Deploy callback feature for phone/chat queues"
            ),
            "effort": "Low",
            "timeline": "2-3 weeks",
            "owner": "Ops Engineering",
            "priority": "P1",
            "category": "📈 Revenue",
        },
        {
            "rank": 5,
            "title": "First-Contact Resolution Improvement",
            "impact_usd": fcr_savings_annual,
            "current": f"Avg contacts/ticket: {avg_contacts:.1f} | Multi-contact: {multi_contact_pct:.0%}",
            "target": "Reduce multi-contact rate by 15%",
            "root_cause": (
                f"Avg {avg_contacts:.1f} contacts per ticket. {multi_contact_pct:.0%} of tickets "
                f"require more than 1 contact. Each re-contact costs ~${kpi['avg_cost_usd']:.2f}. "
                f"Root cause: incomplete information gathering on first contact."
            ),
            "solution": (
                "1) Implement AI-powered agent assist with suggested responses\n"
                "2) Build dynamic ticket forms that pre-gather context by category\n"
                "3) Create resolution checklists for top 10 ticket types"
            ),
            "effort": "High",
            "timeline": "8-12 weeks",
            "owner": "Product + Ops",
            "priority": "P2",
            "category": "⚙️ Process",
        },
    ]
    return opportunities


def render_opportunities_tab(
    df: pd.DataFrame, analytics: dict, nlp_results: dict
) -> None:
    """Render the opportunities identification tab."""
    st.subheader("🎯 Top 5 Improvement Opportunities")
    st.caption(
        f"All impact estimates extrapolated using {SCALE_FACTOR}× scale factor "
        "(sample → Groupon monthly volume)"
    )

    opportunities = _build_opportunities(analytics, nlp_results, df)

    total_impact = sum(o["impact_usd"] for o in opportunities)
    st.markdown(
        f"**Combined estimated annual impact: "
        f"${total_impact:,.0f}**"
    )
    st.markdown("---")

    for opp in opportunities:
        emoji = {"P0": "🔴", "P1": "🟡", "P2": "🟢"}.get(opp["priority"], "⚪")
        with st.expander(
            f"{emoji} #{opp['rank']} — {opp['title']}  |  "
            f"~${opp['impact_usd']:,.0f}/yr  |  {opp['priority']}",
            expanded=opp["rank"] == 1,
        ):
            col1, col2, col3 = st.columns(3)
            col1.metric("Current", opp["current"])
            col2.metric("Target", opp["target"])
            col3.metric("Category", opp["category"])

            st.markdown(f"**Root Cause:** {opp['root_cause']}")
            st.markdown(f"**AI-First Solution:**\n{opp['solution']}")

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Effort:** {opp['effort']}")
            c2.markdown(f"**Timeline:** {opp['timeline']}")
            c3.markdown(f"**Owner:** {opp['owner']}")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — Trends
# ──────────────────────────────────────────────────────────────────────────────
def render_trends_tab(df: pd.DataFrame, analytics: dict) -> None:
    """Render the weekly trends analysis tab."""
    st.subheader("📊 Weekly Trends (Weeks 7-10)")

    # Derived metric trend selector
    derived_df = pd.DataFrame(analytics["derived_trends"])
    if not derived_df.empty:
        derived_metric = st.selectbox(
            "Derived KPI",
            [
                "resolution_rate",
                "escalation_rate",
                "abandonment_rate",
                "avg_csat",
                "avg_frt",
                "avg_resolution_min",
                "avg_cost",
                "ticket_count",
            ],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        fig = plot_kpi_trend(
            derived_df,
            derived_metric,
            f"{derived_metric.replace('_', ' ').title()} — Weekly Trend",
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # Raw metric trends
    raw_df = pd.DataFrame(analytics["raw_trends"])
    if not raw_df.empty:
        raw_metric = st.selectbox(
            "Raw metric deep-dive",
            [
                "csat_score",
                "first_response_min",
                "resolution_min",
                "cost_usd",
                "contacts_per_ticket",
            ],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        fig = plot_kpi_trend(
            raw_df,
            raw_metric,
            f"{raw_metric.replace('_', ' ').title()} — Weekly Avg",
        )
        st.plotly_chart(fig, width="stretch")

    # Team comparison
    st.markdown("### Team Comparison")
    team_df = pd.DataFrame(analytics["teams"])
    team_metric = st.selectbox(
        "Compare teams by",
        [
            "resolution_rate",
            "avg_csat",
            "avg_cost_usd",
            "avg_frt_min",
            "escalation_rate",
        ],
        format_func=lambda x: x.replace("_", " ").title(),
    )
    fig = plot_team_comparison(
        team_df,
        team_metric,
        f"Team Comparison: {team_metric.replace('_', ' ').title()}",
    )
    st.plotly_chart(fig, width="stretch")

    # Correlation insights
    if analytics.get("correlations"):
        st.markdown("### CSAT Correlation Analysis")
        corr = analytics["correlations"]
        corr_data = []
        for feat, vals in corr.items():
            if isinstance(vals, dict):
                corr_data.append(
                    {
                        "Feature": feat.replace("_", " ").title(),
                        "Correlation": vals.get("correlation", 0),
                        "P-value": vals.get("p_value", 1),
                        "Significant": "✅" if vals.get("p_value", 1) < 0.05 else "❌",
                    }
                )
        if corr_data:
            st.dataframe(pd.DataFrame(corr_data), width="stretch")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 4 — NLP Insights
# ──────────────────────────────────────────────────────────────────────────────
def render_nlp_tab(df: pd.DataFrame, nlp_results: dict) -> None:
    """Render the NLP insights tab."""
    st.subheader("💬 NLP Insights")

    if not nlp_results:
        st.warning("NLP analysis not available.")
        return

    # Sentiment overview
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Sentiment", f"{nlp_results.get('avg_sentiment_polarity', 0):.3f}")
    col2.metric("Frustration Rate", f"{nlp_results.get('frustration_rate', 0):.1%}")
    col3.metric(
        "Frustrated Tickets",
        f"{nlp_results.get('total_frustrated_tickets', 0):,}",
    )

    st.markdown("---")

    # Sentiment distribution
    if nlp_results.get("sentiment_distribution"):
        st.markdown("### Sentiment Distribution")
        # Add sentiment columns so the chart works
        df_with_sent = add_sentiment_columns(df)
        fig = plot_sentiment_distribution(df_with_sent)
        st.plotly_chart(fig, width="stretch")

    # Frustration by category
    if nlp_results.get("frustration_by_category"):
        st.markdown("### Frustration Rate by Category")
        fig = plot_frustration_by_category(nlp_results["frustration_by_category"])
        st.plotly_chart(fig, width="stretch")

    # Topics
    if nlp_results.get("topics"):
        st.markdown("### Detected Topics (TF-IDF Clustering)")
        for i, topic in enumerate(nlp_results["topics"], 1):
            words = ", ".join(topic["top_words"][:7])
            count = topic.get("count", "?")
            st.markdown(f"**Topic {i}** ({count} tickets): {words}")

    # Sample frustrated messages
    if nlp_results.get("sample_frustrated_messages"):
        st.markdown("### Sample Frustrated Messages")
        for msg in nlp_results["sample_frustrated_messages"][:5]:
            st.markdown(f'> *"{msg}"*')


# ──────────────────────────────────────────────────────────────────────────────
# Tab 5 — Weekly Brief
# ──────────────────────────────────────────────────────────────────────────────
def render_weekly_brief_tab(
    analytics: dict,
    nlp_results: dict,
    quality_report: dict,
    cleaning_log: dict,
) -> None:
    """Render the weekly brief tab."""
    st.subheader("📋 Weekly Operations Brief")

    kpi = analytics["kpi"]
    chatbot = analytics["chatbot"]
    teams_list = analytics["teams"]

    # Find best/worst teams
    best_team = max(teams_list, key=lambda t: t.get("avg_csat", 0))
    worst_team = min(teams_list, key=lambda t: t.get("avg_csat", 5))

    # Derived trends for WoW changes
    derived = analytics.get("derived_trends", [])
    wow_text = ""
    if len(derived) >= 2:
        last = derived[-1]
        prev = derived[-2]
        res_delta = last.get("resolution_rate", 0) - prev.get("resolution_rate", 0)
        csat_delta = last.get("avg_csat", 0) - prev.get("avg_csat", 0)
        res_emoji = "🟢" if res_delta >= 0 else "🔴"
        csat_emoji = "🟢" if csat_delta >= 0 else "🔴"
        wow_text = (
            f"| Resolution Rate WoW | {res_emoji} {res_delta:+.1%} |\n"
            f"| CSAT WoW | {csat_emoji} {csat_delta:+.2f} |"
        )

    frustration_rate = nlp_results.get("frustration_rate", 0)
    avg_polarity = nlp_results.get("avg_sentiment_polarity", 0)

    brief = f"""
## 📊 Weekly Ops Intelligence Brief
### Weeks {min(COMPLETE_WEEKS)}–{max(COMPLETE_WEEKS)} | Groupon Customer Operations

---

### Executive Summary

Groupon's customer support operation processed **{kpi['total_tickets']:,} tickets** in this period
(~**{kpi['total_tickets'] * SCALE_FACTOR:,}/month** at scale). The overall resolution rate is
**{kpi['resolution_rate']:.1%}** with an average CSAT of **{kpi['avg_csat']:.2f}/5**. The
**AI chatbot escalation rate of {chatbot['overall_escalation_rate']:.1%}** is the single largest
improvement opportunity, followed by **BPO Vendor B's CSAT gap** ({worst_team.get('avg_csat', 0):.2f}
vs {best_team.get('avg_csat', 0):.2f} in-house).

---

### Key Metrics Dashboard

| Metric | Value | Status |
|--------|-------|--------|
| Total Tickets | {kpi['total_tickets']:,} | — |
| Resolution Rate | {kpi['resolution_rate']:.1%} | {'🟢' if kpi['resolution_rate'] > 0.60 else '🔴'} |
| Escalation Rate | {kpi['escalation_rate']:.1%} | {'🟢' if kpi['escalation_rate'] < 0.15 else '🔴'} |
| Abandonment Rate | {kpi['abandonment_rate']:.1%} | {'🟢' if kpi['abandonment_rate'] < 0.05 else '🟡' if kpi['abandonment_rate'] < 0.10 else '🔴'} |
| Avg First Response | {kpi['avg_first_response_min']:.0f} min | {'🟢' if kpi['avg_first_response_min'] < 30 else '🟡' if kpi['avg_first_response_min'] < 60 else '🔴'} |
| Avg Resolution Time | {kpi['avg_resolution_min']:.0f} min | — |
| Avg CSAT | {kpi['avg_csat']:.2f}/5 | {'🟢' if kpi['avg_csat'] > 3.5 else '🟡' if kpi['avg_csat'] > 3.0 else '🔴'} |
| Avg Cost/Ticket | ${kpi['avg_cost_usd']:.2f} | — |
| CSAT Collection Rate | {kpi.get('csat_collection_rate', 0):.1%} | {'🟡' if kpi.get('csat_collection_rate', 0) < 0.80 else '🟢'} |
{wow_text}

---

### ⚠️ Watch List

- 🔴 **Chatbot Escalations** — {chatbot['overall_escalation_rate']:.1%} escalation rate; merchant_issue and billing categories need routing bypass
- 🔴 **BPO Vendor B Quality** — CSAT {worst_team.get('avg_csat', 0):.2f}, {(best_team.get('avg_csat', 0) - worst_team.get('avg_csat', 0)):.2f} points below best team
- 🟡 **Customer Frustration** — {frustration_rate:.1%} of tickets show frustration signals (avg sentiment {avg_polarity:.3f})
- 🟡 **Abandonment** — {kpi['abandonment_rate']:.1%} abandonment rate = ~{kpi['total_tickets'] * SCALE_FACTOR * kpi['abandonment_rate']:,.0f} lost customers/month at scale
- 🟡 **CSAT Collection** — Only {kpi.get('csat_collection_rate', 0):.1%} of tickets have CSAT scores; missing data skews insights

---

### Data Quality Notes

- Rows processed: {quality_report['total_rows']:,}
- Data completeness: {quality_report['completeness_score']:.1%}
- Fixes applied: {', '.join(f'{k}: {v}' for k, v in cleaning_log.items())}
- Week 11 excluded (partial — 224 tickets only)

---

*Prepared by: AI-Powered Ops Command Center | Groupon Global Customer Operations*
"""
    st.markdown(brief)

    # Download button
    st.download_button(
        label="📥 Download Brief as Markdown",
        data=brief,
        file_name="weekly_ops_brief.md",
        mime="text/markdown",
    )

    st.info(
        "💡 Connect your Google Gemini API key and run the full LangGraph pipeline "
        "for an AI-generated brief with deeper insights and custom recommendations."
    )
