"""Visualization functions using Plotly Express.

All functions return Plotly figure objects. No st.plotly_chart calls here.
Layout and branding are applied via consistent theming.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.config import (
    CHART_COLORS,
    GROUPON_DARK,
    GROUPON_GREEN,
    GROUPON_LIGHT_GREEN,
    TEAM_COLORS,
)


def _apply_branding(fig: go.Figure) -> go.Figure:
    """Apply consistent Groupon branding to a figure."""
    fig.update_layout(
        font=dict(family="Inter, -apple-system, sans-serif", size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        title_font_size=16,
        title_font_color=GROUPON_DARK,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def plot_kpi_trend(
    weekly_df: pd.DataFrame, metric: str, title: str
) -> go.Figure:
    """Line chart showing a KPI over weeks.

    Args:
        weekly_df: DataFrame with 'week_number' and metric column.
        metric: Column name to plot.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    fig = px.line(
        weekly_df,
        x="week_number",
        y=metric,
        title=title,
        markers=True,
        color_discrete_sequence=[GROUPON_GREEN],
    )
    fig.update_traces(line=dict(width=3))
    return _apply_branding(fig)


def plot_team_comparison(
    team_df: pd.DataFrame, metric: str, title: str
) -> go.Figure:
    """Bar chart comparing teams on a metric.

    Args:
        team_df: DataFrame with 'team' and metric columns.
        metric: Column name to plot.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    fig = px.bar(
        team_df,
        x="team",
        y=metric,
        title=title,
        color="team",
        color_discrete_map=TEAM_COLORS,
    )
    return _apply_branding(fig)


def plot_channel_distribution(df: pd.DataFrame) -> go.Figure:
    """Pie chart of ticket distribution by channel.

    Args:
        df: Clean DataFrame with 'channel' column.

    Returns:
        Plotly Figure.
    """
    channel_counts = df["channel"].value_counts().reset_index()
    channel_counts.columns = ["channel", "count"]
    fig = px.pie(
        channel_counts,
        values="count",
        names="channel",
        title="Ticket Distribution by Channel",
        color_discrete_sequence=CHART_COLORS,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _apply_branding(fig)


def plot_category_treemap(df: pd.DataFrame) -> go.Figure:
    """Treemap of ticket volume by category and subcategory.

    Args:
        df: Clean DataFrame.

    Returns:
        Plotly Figure.
    """
    tree_data = (
        df.groupby(["category", "subcategory"])
        .size()
        .reset_index(name="count")
    )
    tree_data = tree_data[tree_data["subcategory"].notna()]

    fig = px.treemap(
        tree_data,
        path=["category", "subcategory"],
        values="count",
        title="Ticket Volume by Category → Subcategory",
        color_discrete_sequence=CHART_COLORS,
    )
    return _apply_branding(fig)


def plot_resolution_funnel(df: pd.DataFrame) -> go.Figure:
    """Funnel chart showing ticket resolution flow.

    Args:
        df: Clean DataFrame.

    Returns:
        Plotly Figure.
    """
    total = len(df)
    resolved = (df["resolution_status"] == "resolved").sum()
    escalated = (df["resolution_status"] == "escalated").sum()
    abandoned = (df["resolution_status"] == "abandoned").sum()
    pending = (df["resolution_status"] == "pending").sum()

    fig = go.Figure(
        go.Funnel(
            y=["Total Tickets", "Resolved", "Escalated", "Abandoned", "Pending"],
            x=[total, resolved, escalated, abandoned, pending],
            marker=dict(
                color=[GROUPON_GREEN, GROUPON_LIGHT_GREEN, "#FF9800", "#E91E63", "#9E9E9E"]
            ),
        )
    )
    fig.update_layout(title="Ticket Resolution Funnel")
    return _apply_branding(fig)


def plot_heatmap_hourly(df: pd.DataFrame) -> go.Figure:
    """Heatmap of ticket volume by day-of-week and hour.

    Args:
        df: Clean DataFrame with 'day_of_week' and 'hour_of_day' columns.

    Returns:
        Plotly Figure.
    """
    day_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
    pivot = df.pivot_table(
        index="day_of_week",
        columns="hour_of_day",
        values="ticket_id",
        aggfunc="count",
    )
    pivot = pivot.reindex(day_order)

    fig = px.imshow(
        pivot,
        title="Ticket Volume Heatmap (Day × Hour)",
        labels=dict(x="Hour of Day", y="Day of Week", color="Tickets"),
        color_continuous_scale=[
            [0, "white"],
            [0.5, GROUPON_LIGHT_GREEN],
            [1, GROUPON_GREEN],
        ],
        aspect="auto",
    )
    return _apply_branding(fig)


def plot_sentiment_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of sentiment polarity scores.

    Args:
        df: DataFrame with 'sentiment_polarity' column.

    Returns:
        Plotly Figure.
    """
    fig = px.histogram(
        df,
        x="sentiment_polarity",
        nbins=40,
        title="Sentiment Polarity Distribution",
        color_discrete_sequence=[GROUPON_GREEN],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Neutral")
    return _apply_branding(fig)


def plot_frustration_by_category(
    frustration_data: dict,
) -> go.Figure:
    """Bar chart of frustration rate by ticket category.

    Args:
        frustration_data: Dict mapping category → frustration rate.

    Returns:
        Plotly Figure.
    """
    df = pd.DataFrame(
        list(frustration_data.items()),
        columns=["category", "frustration_rate"],
    )
    df = df.sort_values("frustration_rate", ascending=True)

    fig = px.bar(
        df,
        x="frustration_rate",
        y="category",
        orientation="h",
        title="Customer Frustration Rate by Category",
        color_discrete_sequence=["#E91E63"],
    )
    fig.update_layout(xaxis_tickformat=".0%")
    return _apply_branding(fig)


def plot_cost_efficiency_scatter(team_df: pd.DataFrame) -> go.Figure:
    """Scatter plot of cost vs. resolution rate by team.

    Args:
        team_df: Team performance DataFrame.

    Returns:
        Plotly Figure.
    """
    fig = px.scatter(
        team_df,
        x="avg_cost_usd",
        y="resolution_rate",
        size="ticket_count",
        color="team",
        color_discrete_map=TEAM_COLORS,
        title="Cost Efficiency: Cost vs. Resolution Rate",
        hover_data=["avg_csat", "escalation_rate"],
    )
    fig.update_layout(
        xaxis_title="Avg Cost per Ticket ($)",
        yaxis_title="Resolution Rate",
        yaxis_tickformat=".0%",
    )
    return _apply_branding(fig)


def create_kpi_card(label: str, value: str, delta: str | None = None) -> str:
    """Generate HTML for a KPI card (for use in Streamlit st.markdown).

    Args:
        label: KPI label text.
        value: Main value display.
        delta: Optional change indicator (e.g., "+5.2%").

    Returns:
        HTML string for the KPI card.
    """
    delta_html = ""
    if delta:
        color = GROUPON_GREEN if not delta.startswith("-") else "#E91E63"
        delta_html = f'<p style="color:{color};font-size:14px;margin:0">{delta}</p>'

    return f"""
    <div style="
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        border-left: 4px solid {GROUPON_GREEN};
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    ">
        <p style="color:#666;font-size:12px;margin:0;text-transform:uppercase">{label}</p>
        <p style="color:{GROUPON_DARK};font-size:28px;font-weight:700;margin:4px 0">{value}</p>
        {delta_html}
    </div>
    """
