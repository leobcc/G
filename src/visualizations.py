"""Visualization functions using Plotly Express.

All functions return Plotly figure objects. No st.plotly_chart calls here.
Layout and branding are applied via consistent theming.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.config import (
    CHART_COLORS,
    GROUPON_GREEN,
    GROUPON_LIGHT_GREEN,
    HEATMAP_SCALE,
    SENTIMENT_COLORS,
    TEAM_COLORS,
)
from plotly.subplots import make_subplots


# ── Warm-theme palette (matches .streamlit/config.toml) ──────────────────────
_BG_CREAM = "#fdfdf8"
_TEXT_DARK = "#3d3a2a"
_GRID_COLOR = "#ecebe3"
_BORDER_COLOR = "#d3d2ca"


def _apply_branding(fig: go.Figure) -> go.Figure:
    """Apply consistent warm branding to a figure."""
    update: dict[str, object] = dict(
        font=dict(family="sans-serif", size=12, color=_TEXT_DARK),
        plot_bgcolor=_BG_CREAM,
        paper_bgcolor=_BG_CREAM,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
    )
    # Only set title font when a title exists to avoid "undefined" rendering
    existing_title = getattr(fig.layout.title, "text", None) if fig.layout.title else None
    if existing_title:
        update["title_font_size"] = 16
        update["title_font_color"] = _TEXT_DARK
    fig.update_layout(**update)
    fig.update_xaxes(showgrid=True, gridcolor=_GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID_COLOR)
    return fig


def plot_kpi_trend(
    weekly_df: pd.DataFrame,
    metric: str,
    title: str,
    week_date_ranges: dict[int, str] | None = None,
) -> go.Figure:
    """Line chart showing a KPI over weeks.

    Args:
        weekly_df: DataFrame with 'week_number' and metric column.
        metric: Column name to plot.
        title: Chart title.
        week_date_ranges: Optional mapping of week number to date range string.

    Returns:
        Plotly Figure.
    """
    plot_df = weekly_df.copy()
    if week_date_ranges:
        plot_df["week_label"] = plot_df["week_number"].apply(
            lambda w: f"Wk {int(w)}\n({week_date_ranges.get(int(w), '')})"
        )
        x_col = "week_label"
    else:
        x_col = "week_number"

    fig = px.line(
        plot_df,
        x=x_col,
        y=metric,
        title=title,
        markers=True,
        color_discrete_sequence=[GROUPON_GREEN],
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.update_xaxes(title_text="")
    # Apply context-aware y-axis formatting
    if "cost" in metric:
        fig.update_yaxes(tickprefix="$")
    elif "min" in metric:
        fig.update_yaxes(ticksuffix=" min")
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
        text_auto=".1%",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(tickangle=-15)
    fig.update_layout(bargap=0.3, height=400, yaxis_range=[0, team_df[metric].max() * 1.15])
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
            textinfo="value+percent initial",
            marker=dict(
                color=[GROUPON_GREEN, GROUPON_LIGHT_GREEN, CHART_COLORS[2], CHART_COLORS[5], "#8C8C8C"]
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
            [0, _BG_CREAM],
            [0.5, GROUPON_LIGHT_GREEN],
            [1, GROUPON_GREEN],
        ],
        aspect="auto",
    )
    return _apply_branding(fig)


def plot_sentiment_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart of sentiment label distribution.

    Shows the count of five sentiment levels: very_negative, negative,
    neutral, positive, very_positive. Falls back to a polarity histogram
    if ``sentiment_label`` column is not present.

    Args:
        df: DataFrame with 'sentiment_label' (or 'sentiment_polarity') column.

    Returns:
        Plotly Figure.
    """
    if "sentiment_label" in df.columns:
        label_order = ["very_negative", "negative", "neutral", "positive", "very_positive"]
        counts = df["sentiment_label"].value_counts().reindex(label_order, fill_value=0)
        color_map = {
            "very_negative": "#8B2D35",
            "negative": SENTIMENT_COLORS["negative"],
            "neutral": SENTIMENT_COLORS["neutral"],
            "positive": GROUPON_LIGHT_GREEN,
            "very_positive": SENTIMENT_COLORS["positive"],
        }
        fig = go.Figure(
            go.Bar(
                x=counts.index.str.title(),
                y=counts.values,
                marker_color=[color_map.get(label, GROUPON_GREEN) for label in counts.index],
                text=counts.values,
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Sentiment Distribution",
            xaxis_title="Sentiment",
            yaxis_title="Ticket Count",
            height=350,
        )
        return _apply_branding(fig)

    # Fallback: polarity histogram
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
        color_discrete_sequence=[CHART_COLORS[5]],
        text=df["frustration_rate"].apply(lambda x: f"{x:.1%}"),
    )
    fig.update_traces(textposition="outside")
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
        text="team",
    )
    fig.update_traces(textposition="top right", textfont_size=9)
    fig.update_layout(
        xaxis_title="Avg Cost per Ticket ($)",
        xaxis_tickprefix="$",
        yaxis_title="Resolution Rate",
        yaxis_tickformat=".0%",
        height=420,
        margin=dict(l=50, r=80, t=60, b=50),
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
        color = GROUPON_GREEN if not delta.startswith("-") else "#c44e3b"
        delta_html = f'<p style="color:{color};font-size:13px;margin:0">{delta}</p>'

    return f"""
    <div style="
        background: {_BG_CREAM};
        border-left: 3px solid {GROUPON_GREEN};
        border-radius: 0.75rem;
        padding: 14px 16px;
        text-align: center;
    ">
        <p style="color:#7a7868;font-size:11px;margin:0;text-transform:uppercase;
                  letter-spacing:0.04em">{label}</p>
        <p style="color:{_TEXT_DARK};font-size:26px;font-weight:600;margin:4px 0">{value}</p>
        {delta_html}
    </div>
    """


def plot_multi_trend(
    derived_df: pd.DataFrame,
    week_date_ranges: dict[int, str] | None = None,
) -> go.Figure:
    """Small-multiples chart showing key derived KPIs over weeks.

    Args:
        derived_df: DataFrame with 'week_number' and derived metric columns.
        week_date_ranges: Optional mapping of week number to date range string.

    Returns:
        Plotly Figure with 2x2 subplots.
    """
    metrics = [
        ("resolution_rate", "Resolution Rate", ".1%"),
        ("escalation_rate", "Escalation Rate", ".1%"),
        ("avg_csat", "Avg CSAT", ".2f"),
        ("avg_cost", "Avg Cost ($)", "$.2f"),
    ]
    # Filter to metrics that exist in the dataframe
    metrics = [(col, label, fmt) for col, label, fmt in metrics if col in derived_df.columns]
    n = len(metrics)
    if n == 0:
        fig = go.Figure()
        fig.add_annotation(text="No trend data available", showarrow=False)
        return _apply_branding(fig)

    # Build x-axis labels: "Wk 7 (Feb 9-15)" or just week numbers
    if week_date_ranges:
        x_labels = [
            f"Wk {int(w)}\n({week_date_ranges.get(int(w), '')})"
            for w in derived_df["week_number"]
        ]
    else:
        x_labels = [f"Wk {int(w)}" for w in derived_df["week_number"]]

    rows = (n + 1) // 2
    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=[label for _, label, _ in metrics],
        vertical_spacing=0.22,
        horizontal_spacing=0.12,
    )

    colors = [CHART_COLORS[0], CHART_COLORS[1], CHART_COLORS[2], CHART_COLORS[3]]
    for i, (col, label, fmt) in enumerate(metrics):
        r = i // 2 + 1
        c = i % 2 + 1
        y_vals = derived_df[col]
        # Build hover text with proper formatting
        if fmt == ".1%":
            text_vals = [f"{v:.1%}" for v in y_vals]
        elif fmt == "$.2f":
            text_vals = [f"${v:.2f}" for v in y_vals]
        else:
            text_vals = [f"{v:.2f}" for v in y_vals]

        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=y_vals,
                mode="lines+markers+text",
                name=label,
                line=dict(width=2.5, color=colors[i % len(colors)]),
                marker=dict(size=8),
                text=text_vals,
                textposition="top center",
                textfont=dict(size=10),
                showlegend=False,
            ),
            row=r, col=c,
        )
        # Format y-axis per metric
        if fmt == ".1%":
            fig.update_yaxes(tickformat=".0%", row=r, col=c)
        elif fmt == "$.2f":
            fig.update_yaxes(tickprefix="$", tickformat=".2f", row=r, col=c)
        # Label x-axis
        fig.update_xaxes(title_text="", tickangle=-20, row=r, col=c)

    fig.update_layout(height=520, margin=dict(l=50, r=30, t=60, b=50))
    return _apply_branding(fig)


def plot_team_summary_table(team_df: pd.DataFrame) -> go.Figure:
    """Compact Plotly table showing key team metrics side-by-side.

    Args:
        team_df: Team performance DataFrame with metric columns.

    Returns:
        Plotly Figure with a styled table.
    """
    display_cols = {
        "team": "Team",
        "resolution_rate": "Res. Rate",
        "escalation_rate": "Esc. Rate",
        "avg_csat": "CSAT",
        "avg_frt_min": "Avg FRT (min)",
        "avg_cost_usd": "Cost/Ticket",
        "ticket_count": "Tickets",
    }
    available = [c for c in display_cols if c in team_df.columns]
    sub = team_df[available].copy()

    # Format numeric values
    formatters = {
        "resolution_rate": lambda x: f"{x:.1%}",
        "escalation_rate": lambda x: f"{x:.1%}",
        "avg_csat": lambda x: f"{x:.2f}",
        "avg_frt_min": lambda x: f"{x:.0f}",
        "avg_cost_usd": lambda x: f"${x:.2f}",
        "ticket_count": lambda x: f"{int(x):,}",
    }
    for col, fmt in formatters.items():
        if col in sub.columns:
            sub[col] = sub[col].apply(fmt)

    headers = [display_cols.get(c, c) for c in available]
    cell_values = [sub[c].tolist() for c in available]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=headers,
                    fill_color="#ecebe3",
                    font=dict(size=13, color=_TEXT_DARK),
                    align="left",
                    line_color=_BORDER_COLOR,
                ),
                cells=dict(
                    values=cell_values,
                    fill_color=_BG_CREAM,
                    font=dict(size=12, color=_TEXT_DARK),
                    align="left",
                    height=32,
                    line_color=_BORDER_COLOR,
                ),
            )
        ]
    )
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=_BG_CREAM,
    )
    return fig


def plot_correlation_bar(correlations: dict) -> go.Figure:
    """Horizontal bar chart of CSAT correlations with significance markers.

    Args:
        correlations: Dict mapping feature name to {correlation, p_value, n}.

    Returns:
        Plotly Figure.
    """
    features = []
    corr_vals = []
    colors = []

    for feat, vals in correlations.items():
        if isinstance(vals, dict):
            features.append(feat.replace("_", " ").title())
            corr_vals.append(vals["correlation"])
            sig = vals.get("p_value", 1) < 0.05
            colors.append(GROUPON_GREEN if sig else "#9E9E9E")
        else:
            # Legacy format: plain float
            features.append(feat.replace("_", " ").title())
            corr_vals.append(vals)
            colors.append(GROUPON_GREEN)

    fig = go.Figure(
        go.Bar(
            x=corr_vals,
            y=features,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in corr_vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="CSAT Correlation Analysis",
        xaxis_title="Pearson Correlation with CSAT",
        yaxis_title="",
        height=280,
        xaxis=dict(range=[-1, 1], zeroline=True, zerolinecolor=_BORDER_COLOR),
    )
    return _apply_branding(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Chatbot escalation chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_chatbot_escalation_by_category(chatbot_data: dict) -> go.Figure:
    """Horizontal bar chart of chatbot escalation rate by category.

    Args:
        chatbot_data: Dict from compute_chatbot_escalation_analysis().

    Returns:
        Plotly Figure.
    """
    by_cat = chatbot_data.get("by_category", [])
    if not by_cat:
        fig = go.Figure()
        fig.add_annotation(text="No chatbot escalation data", showarrow=False)
        return _apply_branding(fig)

    df = pd.DataFrame(by_cat).sort_values("escalation_rate", ascending=True)

    fig = px.bar(
        df,
        x="escalation_rate",
        y="category",
        orientation="h",
        title="Chatbot Escalation Rate by Category",
        text=df["escalation_rate"].apply(lambda x: f"{x:.1%}"),
        color_discrete_sequence=[CHART_COLORS[2]],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Escalation Rate",
        xaxis_tickformat=".0%",
        yaxis_title="",
        height=320,
    )
    return _apply_branding(fig)


# ──────────────────────────────────────────────────────────────────────────────
# FRT / Resolution distribution box plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_frt_boxplot_by_team(df: pd.DataFrame) -> go.Figure:
    """Box plot of first_response_min distribution by team.

    Args:
        df: Clean DataFrame.

    Returns:
        Plotly Figure.
    """
    fig = px.box(
        df.dropna(subset=["first_response_min"]),
        x="assigned_team",
        y="first_response_min",
        color="assigned_team",
        color_discrete_map=TEAM_COLORS,
        title="First Response Time Distribution by Team",
    )
    fig.update_layout(
        xaxis_title="Team",
        yaxis_title="First Response (min)",
        showlegend=False,
        height=380,
    )
    return _apply_branding(fig)


def plot_resolution_boxplot_by_team(df: pd.DataFrame) -> go.Figure:
    """Box plot of resolution_min distribution by team.

    Args:
        df: Clean DataFrame.

    Returns:
        Plotly Figure.
    """
    valid = df.dropna(subset=["resolution_min"])
    valid = valid[valid["resolution_min"] >= 0]
    fig = px.box(
        valid,
        x="assigned_team",
        y="resolution_min",
        color="assigned_team",
        color_discrete_map=TEAM_COLORS,
        title="Resolution Time Distribution by Team",
    )
    fig.update_layout(
        xaxis_title="Team",
        yaxis_title="Resolution Time (min)",
        showlegend=False,
        height=380,
    )
    return _apply_branding(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CSAT heatmap (category × team)
# ──────────────────────────────────────────────────────────────────────────────

def plot_csat_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of average CSAT by category × team.

    Args:
        df: Clean DataFrame.

    Returns:
        Plotly Figure.
    """
    valid = df[df["csat_score"].between(1, 5)]
    pivot = valid.pivot_table(
        index="category",
        columns="assigned_team",
        values="csat_score",
        aggfunc="mean",
    )
    if pivot.empty:
        fig = go.Figure()
        fig.add_annotation(text="No CSAT data available", showarrow=False)
        return _apply_branding(fig)

    fig = px.imshow(
        pivot.round(2),
        title="Average CSAT by Category × Team",
        labels=dict(x="Team", y="Category", color="Avg CSAT"),
        color_continuous_scale=HEATMAP_SCALE,
        aspect="auto",
        text_auto=".2f",
    )
    fig.update_layout(height=360)
    return _apply_branding(fig)


# ──────────────────────────────────────────────────────────────────────────────
# BPO vendor comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_bpo_comparison(team_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing BPO Vendor A vs Vendor B on key metrics.

    Args:
        team_df: Team performance DataFrame (list-of-dicts or DataFrame).

    Returns:
        Plotly Figure.
    """
    if isinstance(team_df, list):
        team_df = pd.DataFrame(team_df)

    bpo = team_df[team_df["team"].str.startswith("bpo_")].copy()
    if bpo.empty:
        fig = go.Figure()
        fig.add_annotation(text="No BPO data available", showarrow=False)
        return _apply_branding(fig)

    metrics = [
        ("resolution_rate", "Resolution Rate", ".1%"),
        ("escalation_rate", "Escalation Rate", ".1%"),
        ("avg_csat", "Avg CSAT", ".2f"),
        ("avg_frt_min", "Avg FRT (min)", ".0f"),
        ("avg_cost_usd", "Cost/Ticket ($)", ".2f"),
    ]

    fig = make_subplots(
        rows=1, cols=len(metrics),
        subplot_titles=[label for _, label, _ in metrics],
        horizontal_spacing=0.08,
    )
    colors = {"bpo_vendorA": TEAM_COLORS["bpo_vendorA"], "bpo_vendorB": TEAM_COLORS["bpo_vendorB"]}

    for i, (col, label, fmt) in enumerate(metrics):
        if col not in bpo.columns:
            continue
        for _, row in bpo.iterrows():
            val = row[col]
            if fmt == ".1%":
                text = f"{val:.1%}"
            elif fmt == ".2f":
                text = f"{val:.2f}"
            else:
                text = f"{val:.0f}"
            fig.add_trace(
                go.Bar(
                    x=[row["team"]],
                    y=[val],
                    name=row["team"],
                    marker_color=colors.get(row["team"], GROUPON_GREEN),
                    text=[text],
                    textposition="inside",
                    textfont=dict(size=10),
                    showlegend=(i == 0),
                ),
                row=1, col=i + 1,
            )

    fig.update_layout(
        title="BPO Vendor Performance Comparison",
        height=400,
        barmode="group",
        margin=dict(l=40, r=40, t=80, b=60),
    )
    # Hide redundant x-axis labels (legend shows team names)
    fig.update_xaxes(showticklabels=False)
    return _apply_branding(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Sentiment cross-dimensional charts
# ──────────────────────────────────────────────────────────────────────────────

def plot_sentiment_by_dimension(
    dimension_data: dict[str, dict],
    dimension_label: str = "Team",
) -> go.Figure:
    """Grouped bar chart of avg polarity & frustration rate by a dimension.

    Args:
        dimension_data: Dict from compute_nlp_summary (e.g. sentiment_by_team).
            Each value is a dict with avg_polarity and frustration_rate.
        dimension_label: Human label for the dimension axis.

    Returns:
        Plotly Figure.
    """
    if not dimension_data:
        fig = go.Figure()
        fig.add_annotation(text=f"No sentiment data by {dimension_label.lower()}", showarrow=False)
        return _apply_branding(fig)

    names = list(dimension_data.keys())
    polarities = [v.get("avg_polarity", 0) for v in dimension_data.values()]
    frustrations = [v.get("frustration_rate", 0) for v in dimension_data.values()]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Avg Sentiment Polarity", "Frustration Rate"],
    )

    fig.add_trace(
        go.Bar(
            x=names, y=polarities, name="Polarity",
            marker_color=GROUPON_GREEN,
            text=[f"{v:.3f}" for v in polarities],
            textposition="outside",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=names, y=frustrations, name="Frustration",
            marker_color=CHART_COLORS[5],
            text=[f"{v:.1%}" for v in frustrations],
            textposition="outside",
        ),
        row=1, col=2,
    )
    fig.update_yaxes(title_text="Polarity", row=1, col=1)
    fig.update_yaxes(title_text="Rate", tickformat=".0%", row=1, col=2)
    fig.update_xaxes(tickangle=-25)
    fig.update_layout(
        title=f"Sentiment Analysis by {dimension_label}",
        height=380,
        showlegend=False,
        margin=dict(l=50, r=30, t=80, b=80),
    )
    return _apply_branding(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Effort-Impact matrix
# ──────────────────────────────────────────────────────────────────────────────

def plot_effort_impact_matrix(opportunities: list[dict]) -> go.Figure:
    """Scatter plot of opportunities on an effort vs. impact matrix.

    Args:
        opportunities: List of opportunity dicts with priority_score, effort,
            title, and impact_estimate fields.

    Returns:
        Plotly Figure.
    """
    if not opportunities:
        fig = go.Figure()
        fig.add_annotation(text="No opportunities to plot", showarrow=False)
        return _apply_branding(fig)

    effort_map = {"low": 1, "medium": 2, "high": 3}
    records = []
    for opp in opportunities:
        effort_str = str(opp.get("effort", "medium")).lower()
        records.append({
            "title": opp.get("title", ""),
            "impact": opp.get("priority_score", 50),
            "effort": effort_map.get(effort_str, 2),
            "effort_label": effort_str.title(),
            "impact_estimate": opp.get("impact_estimate", ""),
        })
    df = pd.DataFrame(records)

    fig = px.scatter(
        df,
        x="effort",
        y="impact",
        text="title",
        size=[40] * len(df),
        color_discrete_sequence=[GROUPON_GREEN],
        title="Effort–Impact Matrix",
        hover_data=["impact_estimate"],
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_layout(
        xaxis=dict(
            title="Effort",
            tickmode="array",
            tickvals=[1, 2, 3],
            ticktext=["Low", "Medium", "High"],
            range=[0.5, 3.5],
        ),
        yaxis=dict(title="Impact Score", range=[0, 105]),
        height=400,
    )
    # Add quadrant guidelines
    fig.add_hline(y=50, line_dash="dash", line_color=_BORDER_COLOR, opacity=0.5)
    fig.add_vline(x=2, line_dash="dash", line_color=_BORDER_COLOR, opacity=0.5)
    return _apply_branding(fig)
