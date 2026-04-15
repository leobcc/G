# Agent Instructions: AI-Powered Customer Operations Command Center

## MISSION

You are implementing an **AI-Powered Customer Operations Command Center** for a Groupon Chief of Staff interview business case. The system ingests customer support ticket data, runs automated multi-step analytics, and produces a weekly "Ops Intelligence Brief" with the top 5 improvement opportunities.

**This must be production-quality work that demonstrates:**
1. Analytical depth and rigor
2. AI-first thinking (not AI as afterthought)
3. Working software (not just a deck)
4. Strategic clarity connecting operations to business impact
5. Professional communication

---

## CRITICAL RULES

### Code Standards
- Python 3.11+. Type hints on all functions. Docstrings on all public functions.
- Use `ruff` for linting. Follow PEP 8.
- All data processing uses pandas. All charts use Plotly Express.
- No hardcoded file paths — use `config.py` for all paths and constants.
- Handle errors gracefully — never let the app crash on bad data.

### Data Rules
- **NEVER modify the original CSV.** Always load from `data/option_a_ticket_data.csv` and clean in memory.
- **All data cleaning must be documented** — every transformation logged so we can explain it.
- **Scale factor: 12x.** The sample has ~10K tickets over 4 weeks. Groupon handles 120K/month. All impact estimates must be calculated on the sample first, then extrapolated with 12x, with explicit uncertainty ranges.
- **Week 11 is partial** (only 224 tickets). Exclude from week-over-week comparisons or note the incompleteness. Use Weeks 7-10 as the four complete weeks.

### LLM/Agent Rules
- Use **Claude** via `langchain-anthropic`. Model: `claude-sonnet-4-20250514` for analysis nodes, `claude-haiku-3-20240307` for simple formatting tasks.
- All LLM outputs use **Pydantic structured output** — never parse free-text LLM responses.
- API key loaded from `.env` file via `python-dotenv`. Never hardcode keys.
- Tool functions must be **pure Python** with no LLM calls inside them — tools compute, the agent reasons.
- Each LangGraph node should have clear, single responsibility.

### Presentation Rules
- Groupon brand color: `#53A318` (green). Use as accent throughout.
- All charts must have clear titles, axis labels, and legends.
- Numbers should always have context: "$5.72 per ticket" not "$5.72".
- Percentages to 1 decimal place. Dollars to 2 decimal places or whole numbers for large amounts.

---

## PROJECT STRUCTURE

```
G/
├── .env                          # ANTHROPIC_API_KEY=sk-...
├── .env.example                  # ANTHROPIC_API_KEY=your-key-here
├── .gitignore
├── README.md
├── requirements.txt
├── AI_COCC_implementation_plan.md
├── AGENT_INSTRUCTIONS.md
│
├── data/
│   └── option_a_ticket_data.csv
│
├── docs/
│   ├── Best_business_case.md
│   └── CoS_Business_Case_Assignment.md
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_cleaning.py
│   ├── analytics.py
│   ├── nlp_analysis.py
│   ├── visualizations.py
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── tools.py
│   │   ├── nodes.py
│   │   ├── graph.py
│   │   └── prompts.py
│   │
│   └── app/
│       ├── streamlit_app.py
│       ├── components.py
│       └── styles.py
│
├── notebooks/
│   └── eda_analysis.ipynb
│
├── output/
│
└── tests/
    ├── test_data_cleaning.py
    ├── test_analytics.py
    └── test_agent.py
```

---

## IMPLEMENTATION SEQUENCE

### Phase 1: Data Foundation (`src/config.py`, `src/data_cleaning.py`, `src/analytics.py`)

#### `src/config.py`
```python
"""Central configuration for the Ops Intelligence system."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
RAW_DATA_PATH = DATA_DIR / "option_a_ticket_data.csv"

# Data constants
VALID_CHANNELS = ["email", "chat", "phone", "social"]
VALID_CATEGORIES = ["refund", "order_status", "merchant_issue", "voucher_problem", "billing", "account", "other"]
VALID_PRIORITIES = ["low", "medium", "high", "urgent"]
VALID_TEAMS = ["in_house", "bpo_vendorA", "bpo_vendorB", "ai_chatbot"]
VALID_STATUSES = ["resolved", "escalated", "abandoned", "pending"]
VALID_MARKETS = ["US", "UK", "DE", "FR", "ES", "IT", "AU"]

MARKET_NORMALIZATION = {
    "United Kingdom": "UK",
    "GER": "DE",
    "USA": "US",
    "": "Unknown",
}

CSAT_MIN = 1
CSAT_MAX = 5

# Scale factor for extrapolation
SCALE_FACTOR = 12  # 10K sample ≈ 1 month segment, Groupon does 120K/month

# Complete weeks in the dataset (exclude partial Week 11)
COMPLETE_WEEKS = [7, 8, 9, 10]

# Groupon branding
GROUPON_GREEN = "#53A318"
GROUPON_DARK = "#1A1A1A"

# LLM settings
LLM_MODEL_ANALYSIS = "claude-sonnet-4-20250514"
LLM_MODEL_SIMPLE = "claude-haiku-3-20240307"
```

#### `src/data_cleaning.py` — DETAILED SPECIFICATION

This module handles ALL data cleaning. It must:

1. **`load_raw_data(path) -> pd.DataFrame`**: Load CSV, return raw DataFrame. Log row count.

2. **`clean_data(df) -> tuple[pd.DataFrame, dict]`**: Apply all cleaning steps, return (clean_df, cleaning_log).

   Cleaning steps (in order):
   - **Normalize market labels**: Apply `MARKET_NORMALIZATION` map. Count affected rows.
   - **Clamp CSAT**: Set values < 1 or > 5 to NaN. Count affected rows.
   - **Fix negative resolution_min**: Set values < 0 to NaN. Count affected rows.
   - **Parse timestamps**: Convert `created_at` to datetime.
   - **Add derived columns**:
     - `week_number`: ISO week number
     - `day_of_week`: Monday, Tuesday, etc.
     - `hour_of_day`: 0-23
     - `is_resolved`: boolean (resolution_status == 'resolved')
     - `is_business_hours`: boolean (hour 8-18, Mon-Fri)
   - **Return cleaning_log**: dict with counts of each transformation applied

3. **`get_data_quality_report(raw_df, clean_df) -> dict`**: Generate a comprehensive data quality report.
   Return structure:
   ```python
   {
       "total_rows": int,
       "total_columns": int,
       "date_range": {"start": str, "end": str},
       "missing_values": {col: {"count": int, "pct": float}},
       "dirty_values_fixed": {
           "market_normalized": int,
           "csat_clamped": int,
           "negative_resolution_fixed": int,
       },
       "completeness_score": float,  # overall % of non-null values
   }
   ```

#### `src/analytics.py` — DETAILED SPECIFICATION

This module provides all statistical analysis functions. Each function takes a clean DataFrame and returns a dict or DataFrame.

Required functions:

1. **`compute_kpi_summary(df, week=None) -> dict`**: Overall KPIs. If week specified, filter to that week.
   ```python
   {
       "total_tickets": int,
       "avg_first_response_min": float,
       "median_first_response_min": float,
       "avg_resolution_min": float,
       "median_resolution_min": float,
       "resolution_rate": float,
       "escalation_rate": float,
       "abandonment_rate": float,
       "avg_csat": float,
       "csat_collection_rate": float,
       "avg_cost_usd": float,
       "total_cost_usd": float,
       "avg_contacts_per_ticket": float,
   }
   ```

2. **`compute_metric_by_dimension(df, metric, dimension, agg='mean') -> pd.DataFrame`**: Group by dimension, compute metric. Return sorted DataFrame.

3. **`compare_weeks(df, metric, week_a, week_b) -> dict`**: Week-over-week comparison for a metric.
   ```python
   {
       "week_a_value": float,
       "week_b_value": float,
       "absolute_change": float,
       "pct_change": float,
       "direction": "improving" | "worsening" | "stable",
   }
   ```

4. **`compute_weekly_trends(df, metrics) -> pd.DataFrame`**: Compute given metrics for each complete week. Return DataFrame with weeks as rows.

5. **`compute_team_performance(df) -> pd.DataFrame`**: Full team performance comparison:
   - Resolution rate, escalation rate, abandonment rate
   - Avg/median FRT, avg/median resolution time
   - Avg CSAT, cost per ticket, cost per resolved ticket
   - Total cost, ticket volume

6. **`compute_channel_performance(df) -> pd.DataFrame`**: Same metrics as team performance, grouped by channel.

7. **`compute_category_performance(df) -> pd.DataFrame`**: Same metrics grouped by category.

8. **`find_statistical_outliers(df, column, method='iqr', threshold=1.5) -> pd.DataFrame`**: Return rows that are outliers.

9. **`compute_cost_per_resolved(df, group_by) -> pd.DataFrame`**: Total cost / resolved count, grouped by dimension.

10. **`compute_chatbot_escalation_analysis(df) -> dict`**: Deep dive into chatbot performance:
    - Escalation rate by category and subcategory
    - Categories where chatbot performs worst
    - Volume of tickets that COULD be auto-resolved (based on category + priority patterns)

11. **`run_correlation_analysis(df, target, features) -> dict`**: Pearson correlation + p-values between target (e.g., CSAT) and features (FRT, resolution time, contacts, etc.).

### Phase 2: NLP Analysis (`src/nlp_analysis.py`)

Required functions:

1. **`analyze_sentiment(messages: pd.Series) -> pd.DataFrame`**: TextBlob polarity (-1 to 1) and subjectivity (0 to 1) for each message. Return DataFrame with columns: `message`, `polarity`, `subjectivity`, `sentiment_label` (positive/neutral/negative based on polarity thresholds: >0.1 = positive, <-0.1 = negative, else neutral).

2. **`detect_frustration(messages: pd.Series) -> pd.DataFrame`**: Flag messages containing frustration indicators. Return DataFrame with `message`, `is_frustrated` (bool), `frustration_signals` (list of matched keywords).

   Frustration keywords/patterns:
   ```python
   FRUSTRATION_PATTERNS = [
       r"ridiculous", r"terrible", r"worst", r"horrible", r"unacceptable",
       r"still waiting", r"3rd time|third time", r"no one helps",
       r"never again", r"waste of", r"scam", r"fraud", r"stolen",
       r"rip.?off", r"\b(ugh+|argh+)\b", r"!!+", r"\?\?+",
       r"this is (a )?joke", r"can't believe", r"extremely frustrated",
       r"disgusted", r"furious", r"livid", r"please help",
   ]
   ```

3. **`extract_topics(messages: pd.Series, n_topics=10, n_words=5) -> dict`**: TF-IDF + KMeans clustering.
   Return:
   ```python
   {
       "topics": [
           {"id": int, "keywords": [str], "message_count": int, "example_messages": [str]},
       ],
       "dominant_terms": [(term, tfidf_score)],  # top 20 terms
   }
   ```

4. **`aggregate_nlp_by_dimension(df_with_sentiment, dimension) -> pd.DataFrame`**: Group sentiment scores by a dimension (channel, team, category). Return avg polarity, frustration rate per group.

5. **`detect_emerging_topics(messages_current_week, messages_prior_week, n_topics=5) -> list`**: Compare TF-IDF term frequencies between two weeks. Return terms that increased most in frequency — these are "emerging" topics.

### Phase 3: Visualizations (`src/visualizations.py`)

All functions return a `plotly.graph_objects.Figure`. Each function must:
- Set appropriate title, axis labels, legend
- Use Groupon green (`#53A318`) as primary color where appropriate
- Use a clean, professional color palette
- Return the figure (don't call `fig.show()`)

Required chart functions:

1. **`create_kpi_cards_data(current_kpis, prior_kpis) -> list[dict]`**: NOT a chart — returns data for Streamlit metric cards. Each dict: `{label, value, delta, delta_color}`.

2. **`create_volume_by_channel_chart(df) -> Figure`**: Stacked bar chart of ticket volume by channel per week.

3. **`create_volume_by_category_chart(df) -> Figure`**: Horizontal bar chart of ticket volume by category.

4. **`create_team_performance_radar(team_perf_df) -> Figure`**: Radar/spider chart comparing teams on normalized metrics (FRT, resolution rate, CSAT, cost efficiency). Normalize each metric to 0-1 scale where higher = better.

5. **`create_frt_distribution_boxplot(df) -> Figure`**: Box plots of first_response_min by team.

6. **`create_csat_heatmap(df) -> Figure`**: Heatmap of average CSAT by category × team.

7. **`create_weekly_trend_lines(weekly_trends_df) -> Figure`**: Multi-line chart with subplots for each key metric over weeks.

8. **`create_resolution_status_sunburst(df) -> Figure`**: Sunburst chart: outer ring = resolution_status, inner ring = category.

9. **`create_cost_breakdown_treemap(df) -> Figure`**: Treemap of total cost by team → category.

10. **`create_sentiment_distribution(sentiment_df) -> Figure`**: Histogram of sentiment polarity scores, colored by sentiment_label.

11. **`create_channel_cost_efficiency(df) -> Figure`**: Grouped bar chart: cost per ticket vs. cost per resolved ticket, by channel.

12. **`create_anomaly_scatter(df, anomaly_flags) -> Figure`**: Scatter plot with anomalies highlighted in red.

### Phase 4: LangGraph Agent (`src/agent/`)

#### `src/agent/state.py`

Define the state using TypedDict. **Important**: pandas DataFrames cannot be directly serialized in LangGraph state. Instead, store data as dicts/lists that can be serialized, and convert to DataFrames within nodes.

```python
from typing import TypedDict, Any

class OpsIntelligenceState(TypedDict):
    # Input
    csv_path: str
    target_week: int
    
    # Data (stored as serializable dicts)
    raw_data_records: list[dict]  # df.to_dict('records')
    clean_data_records: list[dict]
    data_quality_report: dict
    cleaning_log: dict
    
    # Analysis outputs
    kpi_current: dict
    kpi_prior: dict
    trend_analysis: dict
    anomaly_flags: list[dict]
    nlp_insights: dict
    team_performance: list[dict]
    channel_performance: list[dict]
    category_performance: list[dict]
    chatbot_analysis: dict
    
    # Synthesis (LLM-generated)
    opportunities: list[dict]
    watch_list: list[dict]
    
    # Output
    weekly_brief: str
    executive_summary: str
    
    # Pipeline metadata
    pipeline_log: list[str]  # log of each step completed
```

#### `src/agent/tools.py`

Define tools that the agent can call. Each tool is a plain Python function decorated with `@tool`. Tools do computation only — no LLM calls inside them.

**IMPORTANT**: Since LangGraph state holds data as dicts, tools should accept dict/list inputs and return dict/list outputs.

Tool list:
```python
from langchain_core.tools import tool

@tool
def compute_metric_by_dimension(records: list[dict], metric: str, 
                                  dimension: str, agg: str = "mean") -> list[dict]:
    """Compute a metric grouped by a dimension (channel, category, team, market, priority).
    Returns list of {dimension_value, metric_value} sorted by metric_value descending."""

@tool
def compare_weeks_metric(records: list[dict], metric: str, 
                          week_a: int, week_b: int) -> dict:
    """Compare a metric between two weeks.
    Returns {week_a_value, week_b_value, absolute_change, pct_change, direction}."""

@tool
def find_outliers(records: list[dict], column: str, method: str = "iqr") -> list[dict]:
    """Find statistical outliers in a numeric column. Returns outlier records."""

@tool
def compute_resolution_cost(records: list[dict], group_by: str) -> list[dict]:
    """Compute cost per resolved ticket grouped by a dimension."""

@tool
def get_chatbot_failure_analysis(records: list[dict]) -> dict:
    """Analyze chatbot escalation patterns by category. Returns {category: escalation_rate}."""

@tool
def get_sentiment_summary(records: list[dict]) -> dict:
    """Run sentiment analysis on customer_message field.
    Returns {avg_polarity, sentiment_distribution, top_frustrated_messages}."""

@tool
def get_weekly_kpi_trend(records: list[dict], metric: str) -> list[dict]:
    """Get week-by-week trend for a metric across complete weeks (7-10).
    Returns [{week, value}]."""
```

#### `src/agent/nodes.py`

Each node function takes `state: OpsIntelligenceState` and returns a partial state update dict.

**Node 1: `ingest_and_clean_node(state)`**
- Load CSV from `state['csv_path']`
- Run `data_cleaning.clean_data()`
- Run `data_cleaning.get_data_quality_report()`
- Store records + quality report in state
- Add to pipeline_log

**Node 2: `data_quality_node(state)`**
- This is an **LLM node**: the LLM reviews the data quality report and decides if any additional checks are needed
- It can call tools (check_completeness, find_outliers) to investigate
- Produces a refined `data_quality_report` with LLM commentary
- This demonstrates agentic behavior: the agent decides what to investigate

**Node 3: `trend_detection_node(state)`** (Parallel)
- Compare target week vs. prior week for all key metrics
- Uses `analytics.compare_weeks()` for each metric
- Produces `trend_analysis` dict
- Deterministic — no LLM needed

**Node 4: `anomaly_detection_node(state)`** (Parallel)
- Run outlier detection on FRT, resolution time, cost
- Cross-dimensional anomalies (e.g., team × category combinations that are way off)
- Produces `anomaly_flags` list
- Deterministic — no LLM needed

**Node 5: `nlp_analysis_node(state)`** (Parallel)
- Run sentiment analysis, frustration detection, topic extraction
- Compare with prior week if available
- Produces `nlp_insights` dict
- Deterministic — no LLM needed

**Node 6: `opportunity_scoring_node(state)`**
- **LLM node**: takes all analysis outputs and synthesizes the top 5 opportunities
- Uses structured output with Pydantic:
  ```python
  class Opportunity(BaseModel):
      rank: int
      title: str
      estimated_annual_impact_usd: float
      current_metric: str
      target_metric: str
      root_cause: str
      ai_first_solution: str
      effort: Literal["low", "medium", "high"]
      timeline_weeks: int
      recommended_action: str
      owner: str
      priority: Literal["P0", "P1", "P2"]
  ```
- Also generates `watch_list`
- System prompt should emphasize: quantify everything, be specific, propose AI-first solutions

**Node 7: `report_generation_node(state)`**
- **LLM node**: takes all state and generates the weekly brief in Markdown format
- Also generates a 3-sentence executive summary
- Use structured output for consistent formatting
- The report must include: executive summary, KPI table with deltas, top 5 opportunities, watch list, data quality notes

#### `src/agent/graph.py`

Assemble the LangGraph:

```python
from langgraph.graph import StateGraph, START, END

def build_ops_intelligence_graph():
    builder = StateGraph(OpsIntelligenceState)
    
    # Add nodes
    builder.add_node("ingest_and_clean", ingest_and_clean_node)
    builder.add_node("data_quality", data_quality_node)
    builder.add_node("trend_detection", trend_detection_node)
    builder.add_node("anomaly_detection", anomaly_detection_node)
    builder.add_node("nlp_analysis", nlp_analysis_node)
    builder.add_node("opportunity_scoring", opportunity_scoring_node)
    builder.add_node("report_generation", report_generation_node)
    
    # Linear start
    builder.add_edge(START, "ingest_and_clean")
    builder.add_edge("ingest_and_clean", "data_quality")
    
    # Parallel analysis after data quality
    builder.add_edge("data_quality", "trend_detection")
    builder.add_edge("data_quality", "anomaly_detection")
    builder.add_edge("data_quality", "nlp_analysis")
    
    # Converge to synthesis
    builder.add_edge("trend_detection", "opportunity_scoring")
    builder.add_edge("anomaly_detection", "opportunity_scoring")
    builder.add_edge("nlp_analysis", "opportunity_scoring")
    
    # Final report
    builder.add_edge("opportunity_scoring", "report_generation")
    builder.add_edge("report_generation", END)
    
    return builder.compile()
```

#### `src/agent/prompts.py`

System prompts for LLM nodes. **Critical** — these prompts shape the quality of the output.

**Data Quality Agent Prompt:**
```
You are a Senior Data Analyst at Groupon reviewing customer support ticket data quality.
You have access to a data quality report and tools to investigate further.

Your job:
1. Review the quality report provided
2. Decide if any additional checks are warranted (use tools if needed)
3. Produce a refined assessment with:
   - Critical issues that affect analysis reliability
   - Non-critical issues worth noting
   - Any data patterns that suggest systemic problems

Be specific. Don't just say "some values are missing" — say "25.7% of CSAT scores are missing, 
concentrated in chatbot-handled tickets, which may indicate the chatbot doesn't prompt for CSAT."
```

**Opportunity Scoring Prompt:**
```
You are the Chief of Staff for Groupon's SVP of Global Operations. You are reviewing 
the weekly analysis of customer support operations to identify the top 5 improvement opportunities.

Context:
- Groupon handles ~120,000 support tickets/month across email, chat, phone, and social
- The data you're analyzing is a sample of ~10,000 tickets over 4 weeks (scale factor: 12x)
- Teams: in-house agents, BPO Vendor A, BPO Vendor B, AI chatbot
- Your audience is the SVP who needs to make resource allocation decisions THIS QUARTER

For each opportunity you MUST:
1. Quantify the annual financial impact (use the 12x scale factor)
2. Explain the root cause with specific data points
3. Propose an AI-first solution (not just "hire more people")
4. Be specific about effort, timeline, and who owns it

Rank opportunities by estimated annual impact (largest first).
Also create a "watch list" of 3-5 emerging patterns that aren't yet critical but need monitoring.

Be data-driven. Every claim must reference specific numbers from the analysis.
```

**Report Generation Prompt:**
```
You are generating the weekly Ops Intelligence Brief for Groupon's customer operations leadership.

Format the report in Markdown with the following sections:
1. Executive Summary (3 sentences maximum, lead with the most important finding)
2. Key Metrics Dashboard (table with This Week, Last Week, Change, Status emoji)
3. Top 5 Opportunities (numbered, each with impact, root cause, solution, action, owner)
4. Watch List (bullet points with ⚠️ emoji)
5. Data Quality Notes (brief, only if relevant)

Style guidelines:
- Use 🟢 for improving, 🟡 for stable, 🔴 for worsening
- Lead each opportunity with the dollar impact
- Keep language crisp and action-oriented — this is for busy executives
- Include specific numbers everywhere
```

### Phase 5: Streamlit App (`src/app/`)

#### `src/app/streamlit_app.py` — MAIN APPLICATION

```python
"""
Groupon Ops Intelligence Command Center
Main Streamlit application.
"""
```

**Page Configuration:**
```python
st.set_page_config(
    page_title="Ops Intelligence Command Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
```

**Sidebar:**
- File uploader (default: load from data/option_a_ticket_data.csv)
- Week selector dropdown (populate from available weeks)
- "🚀 Run Analysis" button (triggers the agent pipeline)
- Expandable "Configuration" section (model selection, etc.)

**Main Area — Tabs:**

**Tab 1: Dashboard Overview**
- Row of 5 KPI metric cards using `st.metric()` with delta values
- Two columns below: volume chart (left), team performance radar (right)
- Full-width: weekly trend line charts

**Tab 2: Opportunities**
- For each opportunity (1-5):
  - `st.expander()` with title + impact amount
  - Inside: root cause, solution, action, effort/timeline
- Impact/effort matrix chart at the bottom

**Tab 3: Trends**
- Week-over-week comparison charts
- Anomaly highlights (data points flagged red)
- Trend direction indicators

**Tab 4: NLP Insights**
- Sentiment distribution chart
- Frustration rate by dimension
- Emerging topics list
- Sample frustrated messages (anonymized)

**Tab 5: Weekly Brief**
- Full Markdown rendered with `st.markdown()`
- Download button `st.download_button()` for Markdown file
- Copy button

**Bottom Section (always visible):**
- Agent pipeline progress log in `st.expander("Agent Activity Log")`
- Shows each step: "✅ Data ingested (10,000 rows)", "✅ Quality checks passed", etc.

#### `src/app/components.py` — REUSABLE COMPONENTS

```python
def render_kpi_card(label, value, delta, delta_color):
    """Render a single KPI metric card."""

def render_opportunity_card(opp: dict):
    """Render an opportunity card with expandable details."""

def render_pipeline_progress(log: list[str], current_step: int, total_steps: int):
    """Render the agent pipeline progress indicator."""

def render_data_quality_badge(quality_report: dict):
    """Render a data quality score badge (green/yellow/red)."""
```

#### `src/app/styles.py` — CUSTOM CSS

```python
CUSTOM_CSS = """
<style>
    .main-header { 
        color: #53A318; 
        font-size: 2.5rem; 
        font-weight: bold; 
    }
    .metric-card { 
        background: #f8f9fa; 
        border-radius: 10px; 
        padding: 1rem; 
        border-left: 4px solid #53A318; 
    }
    .opportunity-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    /* hide streamlit footer */
    footer { visibility: hidden; }
</style>
"""
```

### Phase 6: EDA Notebook (`notebooks/eda_analysis.ipynb`)

This notebook contains the exploratory analysis work for the slide deck. Structure:

1. **Data Loading & Quality** — load, profile, show quality issues
2. **Descriptive Stats** — volume distributions, performance tables
3. **Team Deep Dive** — head-to-head comparison with charts
4. **Channel Analysis** — efficiency by channel
5. **Category Analysis** — which categories drive the most cost
6. **Chatbot Performance** — escalation analysis
7. **NLP Analysis** — sentiment, frustration, topics
8. **Opportunity Sizing** — quantify each opportunity
9. **Export Charts** — save key charts as images for the deck

---

## KEY DATA INSIGHTS TO VALIDATE DURING IMPLEMENTATION

These are preliminary findings from data profiling. Validate and expand during implementation:

1. **Chatbot escalation is the #1 opportunity**: 25.7% escalation rate, highest of any team. Merchant_issue (14.2% of escalations) and billing (13.1%) are categories the chatbot should not be handling.

2. **BPO Vendor B significantly underperforms**: CSAT 3.02 (lowest), FRT 80.3 min (slowest), resolution time 181.6 min (slowest). This is a clear coaching/training/routing problem.

3. **Email channel is a cost sink**: 34.9% of volume but $3.67 avg cost (only phone is higher at $5.35). Avg FRT of 79.2 min is terrible. Many email tickets could be deflected to chat.

4. **In-house agents are the best but most expensive**: CSAT 3.94, 71.9% resolution, but $5.72/ticket. Should be reserved for complex/high-value cases.

5. **8.3% abandonment rate = ~1,000 lost customers/month at scale**: These are customers who gave up. Correlate with FRT — likely the ones who waited too long.

6. **CSAT collection rate is only ~74%**: This itself is an ops issue worth flagging.

7. **Ticket volume is growing**: Week 7: 2,288, Week 8: 2,460, Week 9: 2,530, Week 10: 2,498. Check if this is seasonal or a trend.

---

## TESTING STRATEGY

### `tests/test_data_cleaning.py`
- Test market normalization covers all dirty values
- Test CSAT clamping (edge cases: 0, -1, 6, 10)
- Test negative resolution filtering
- Test derived columns are correctly computed
- Test that cleaning doesn't drop any rows

### `tests/test_analytics.py`
- Test KPI computation with known data (create small fixture)
- Test week comparison with synthetic weeks
- Test outlier detection catches known outliers
- Test cost_per_resolved handles zero-resolved edge case

### `tests/test_agent.py`
- Test that the graph compiles
- Test individual nodes with mock state
- Test end-to-end pipeline produces all required state keys

---

## COMMON PITFALLS TO AVOID

1. **Don't use f-strings with LLM prompts** containing user data — use proper prompt templates to avoid injection
2. **Don't compute cost_per_resolved as total_cost / resolved_count for the WHOLE dataset** — you need to compute it as sum(cost WHERE resolved) / count(resolved). Otherwise you're attributing abandoned/escalated ticket costs to "resolved" metric.
3. **Don't compare Week 11 data** — it's only 224 tickets (partial week). Exclude or clearly note.
4. **Don't show raw sentiment scores without context** — TextBlob polarity on short service messages will cluster near 0. Calibrate expectations.
5. **Don't hardcode specific ticket IDs or agent IDs** — the system should work on any dataset with the same schema.
6. **Cache expensive computations** — use `@st.cache_data` in Streamlit, compute analytics once per data load.
7. **Handle LLM failures gracefully** — if the LLM API is down, show cached/static results rather than crashing.
8. **Don't let the agent run indefinitely** — set max iterations in LangGraph if using agentic loops.

---

## DEFINITION OF DONE

The implementation is complete when:

- [ ] `streamlit run src/app/streamlit_app.py` starts successfully
- [ ] Uploading / loading the CSV triggers the full pipeline
- [ ] All 5 tabs display correctly with interactive charts
- [ ] The agent pipeline completes in < 60 seconds
- [ ] The weekly brief is generated and downloadable
- [ ] Changing the target week produces different results
- [ ] The pipeline log shows clear multi-step reasoning
- [ ] All tests pass
- [ ] Code has no linting errors (ruff)
- [ ] .env.example documents required environment variables
- [ ] README.md has setup and run instructions
