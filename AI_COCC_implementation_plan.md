# AI-Powered Customer Operations Command Center
## Implementation Plan

**Status: IN PROGRESS**

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Data Profile & Key Findings](#2-data-profile--key-findings)
3. [Deliverable 1: Exploratory Analysis & Opportunity Sizing](#3-deliverable-1-exploratory-analysis--opportunity-sizing)
4. [Deliverable 2: Agentic AI Prototype — Weekly Ops Intelligence Agent](#4-deliverable-2-agentic-ai-prototype--weekly-ops-intelligence-agent)
5. [Deliverable 3: Scaling Roadmap](#5-deliverable-3-scaling-roadmap)
6. [Tech Stack](#6-tech-stack)
7. [Project Structure](#7-project-structure)
8. [Presentation & Delivery Strategy](#8-presentation--delivery-strategy)
9. [Risk Mitigation](#9-risk-mitigation)
10. [Timeline](#10-timeline)
11. [Progress Tracker](#11-progress-tracker)

---

## 1. Executive Summary

We are building an **AI-powered Operations Intelligence system** that automatically ingests Groupon's customer support ticket data, runs a multi-step analytical pipeline, and produces a structured weekly "Ops Intelligence Brief" identifying the top 5 improvement opportunities — each quantified by business impact and paired with actionable recommendations.

**Three deliverables:**
1. **Slide deck** (max 15 pages): Exploratory analysis + 5 opportunity cards + methodology
2. **Working agentic prototype** with Streamlit UI: Live demo that runs the full pipeline end-to-end
3. **Scaling roadmap** (1-2 slides): Production evolution plan

---

## 2. Data Profile & Key Findings

### 2.1 Dataset Overview
- **Rows:** 10,000 tickets (simulating ~4 weeks)
- **Date Range:** 2026-02-09 to 2026-03-10 (Weeks 7-11, with Week 11 partial)
- **Columns:** 16 fields (ticket_id, created_at, channel, category, subcategory, priority, customer_message, assigned_team, agent_id, first_response_min, resolution_min, resolution_status, csat_score, contacts_per_ticket, cost_usd, market)

### 2.2 Data Quality Issues Identified
These are INTENTIONAL — handling them is part of the evaluation.

| Issue | Details | Handling Strategy |
|---|---|---|
| **Missing subcategory** | 2,235 rows (22.4%) blank | Impute via NLP on customer_message using keyword matching + LLM classification |
| **Missing CSAT scores** | 2,573 rows (25.7%) null | Analyze only available data, note collection rate as its own insight |
| **Missing resolution_min** | 1,796 rows (18.0%) null — correlates with pending/abandoned status | Keep nulls for unresolved; analyze only resolved tickets for resolution time metrics |
| **Missing agent_id** | 2,790 rows (27.9%) — all are ai_chatbot | Expected behavior, not an error |
| **Missing first_response_min** | 216 rows (2.2%) | Drop for FRT analysis |
| **Dirty market labels** | "United Kingdom" (12), "GER" (11), "USA" (8), blank (11) | Normalize: UK←United Kingdom, GER→DE, USA→US, blank→Unknown |
| **Negative resolution_min** | Min is -107.6 — impossible | Flag as data error, filter out negatives |
| **CSAT out of range** | Min -1, Max 10 — should be 1-5 | Clamp to 1-5 range, or flag out-of-range as anomalies |

### 2.3 Key Statistics

**Volume Distribution:**
- Channels: email (34.9%), chat (29.7%), phone (20.1%), social (15.3%)
- Categories: refund (22.7%), order_status (19.7%), merchant_issue (15.3%), voucher_problem (12.4%), billing (12.0%), account (9.8%), other (8.1%)
- Priority: medium (39.7%), low (25.6%), high (24.6%), urgent (10.1%)
- Teams: in_house (28.9%), ai_chatbot (27.9%), bpo_vendorA (23.9%), bpo_vendorB (19.4%)

**Performance Summary:**
| Team | Resolution Rate | Avg FRT (min) | Avg Resolution (min) | Avg CSAT | Avg Cost | Total Cost |
|---|---|---|---|---|---|---|
| in_house | 71.9% | 32.9 | 78.8 | 3.94 | $5.72 | $16,516 |
| bpo_vendorA | 59.7% | 58.1 | 129.7 | 3.32 | $3.90 | $9,311 |
| bpo_vendorB | 59.1% | 80.3 | 181.6 | 3.02 | $3.30 | $6,393 |
| ai_chatbot | 54.5% | 1.3 | 9.9 | 3.25 | $0.13 | $371 |

**Key Insight:** The chatbot is 40-60x cheaper per ticket, responds 25-60x faster, but has only 54.5% resolution rate with 25.7% escalation — the highest escalation rate of any team. This is the single biggest optimization opportunity.

---

## 3. Deliverable 1: Exploratory Analysis & Opportunity Sizing

### 3.1 Analysis Pipeline (Python Notebook / Scripts)

The analysis will follow this structure and will be presented in the slide deck:

#### Step 1: Data Cleaning & Quality Report
- Load CSV with pandas
- Normalize market names (map "United Kingdom"→UK, "GER"→DE, "USA"→US, blank→"Unknown")
- Clamp CSAT to 1-5 range, flag out-of-range values
- Filter out negative resolution times
- Add derived columns: `week_number`, `day_of_week`, `hour_of_day`, `is_resolved` (boolean)
- Generate data quality summary table for the deck

#### Step 2: Descriptive Analytics
- **Volume analysis**: tickets by channel, category, priority, market, team, week
- **Performance metrics**: FRT, resolution time, resolution rate, CSAT, cost — broken down by every dimension
- **Correlation analysis**: what predicts CSAT? (FRT, resolution time, contacts, channel, category)
- **Time-series trends**: weekly trajectory of key metrics

#### Step 3: Deep-Dive Analytics
- **Chatbot escalation analysis**: which categories/subcategories does the chatbot fail on most?
- **BPO vendor comparison**: head-to-head on every metric
- **Cost efficiency analysis**: cost per resolved ticket (not just per ticket)
- **Channel-mix optimization**: which channels are most efficient for which categories?
- **Market performance**: which markets are underperforming?

#### Step 4: NLP Analysis on Customer Messages
- **Sentiment scoring**: Use TextBlob or a lightweight model to estimate sentiment from customer_message
- **Topic clustering**: TF-IDF + K-means to find natural clusters in messages beyond the category labels
- **Frustration detection**: keyword-based flags ("ridiculous", "terrible", "never", "worst", etc.)
- **Miscategorization detection**: compare NLP-inferred topic vs. assigned category

### 3.2 Top 5 Opportunities (Preliminary)

Based on the data profiling, these are the likely top 5 — final ranking will be confirmed during full analysis:

#### Opportunity 1: Reduce AI Chatbot Escalation Rate (25.7% → target 15%)
- **Prize:** ~300 fewer escalations/month → ~$9,000-12,000/month savings (escalated tickets cost 20-50x more than chatbot-resolved)
- **Root Cause:** Chatbot lacks capability for certain categories (merchant_issue, billing disputes, complex refunds)
- **AI-First Solution:** Implement intent-aware routing — use NLP to pre-classify tickets and only send chatbot-suitable ones to the bot; retrain chatbot on failure patterns
- **Extrapolated to 120K/month:** ~$108K-144K annual savings

#### Opportunity 2: Address BPO Vendor B Performance Gap
- **Prize:** Vendor B has CSAT 3.02 vs. Vendor A 3.32 vs. in-house 3.94; FRT 80.3 min vs. 58.1 vs. 32.9
- **Root Cause:** Likely training gaps, quality monitoring deficiency, or wrong ticket routing
- **AI-First Solution:** AI-powered quality monitoring agent that scores a sample of VendorB interactions in real-time, flags poor responses for supervisor review, and generates weekly coaching reports
- **Impact:** If CSAT improves from 3.02 → 3.32 (+0.30), significant customer retention improvement

#### Opportunity 3: Optimize Channel-Mix Routing by Category
- **Prize:** Email has 79.2 min avg FRT and $3.67 avg cost; chat has 3.7 min FRT and $1.52 cost — but email handles 34.9% of tickets
- **Root Cause:** Ticket routing doesn't consider channel efficiency by category
- **AI-First Solution:** Smart routing system: for categories where chat/chatbot performs well, actively deflect from email/phone to chat; implement proactive chatbot triggers on website for common issues
- **Impact:** Shifting 20% of email volume to chat could save ~$1.50/ticket × 700 tickets = ~$1,050/week → $54K/year extrapolated

#### Opportunity 4: Reduce Abandoned Ticket Rate (8.3%)
- **Prize:** 830 abandoned tickets/month = wasted cost + negative CX; abandoned tickets still incur partial cost
- **Root Cause:** Long wait times leading to customer drop-off (likely correlated with high FRT)
- **AI-First Solution:** Proactive follow-up agent that detects slow-moving tickets and sends automated status updates; implement priority re-scoring for aging tickets
- **Impact:** If 50% of abandoned were retained → improved resolution rate + CSAT

#### Opportunity 5: CSAT Recovery Program for Low-Scoring Interactions
- **Prize:** 25.7% of tickets have no CSAT collected; among collected, ~30% score 1-2
- **Root Cause:** No systematic follow-up on low CSAT; no feedback loop to agents
- **AI-First Solution:** Automated CSAT recovery workflow — when CSAT ≤ 2, trigger AI agent to analyze the interaction, draft a personalized recovery outreach, and alert the team lead
- **Impact:** Improving CSAT collection rate + recovering detractors

### 3.3 Presentation Format for Opportunity Cards

Each opportunity will be presented as a structured card:

```
┌─────────────────────────────────────────────────┐
│  OPPORTUNITY #X: [Title]                        │
│  ───────────────────────────────────────────────│
│  💰 Estimated Annual Impact: $XXX,XXX           │
│  📊 Key Metric: [Current] → [Target]            │
│  ───────────────────────────────────────────────│
│  ROOT CAUSE                                     │
│  [2-3 sentences + supporting data visualization]│
│  ───────────────────────────────────────────────│
│  AI-FIRST SOLUTION                              │
│  [Specific technical approach]                  │
│  ───────────────────────────────────────────────│
│  IMPLEMENTATION                                 │
│  Effort: [Low/Med/High]  Timeline: [X weeks]    │
│  Dependencies: [List]                           │
└─────────────────────────────────────────────────┘
```

### 3.4 Scaling Assumptions for 120K Tickets/Month

The dataset has ~10,000 tickets over ~4 weeks ≈ 2,500/week. Groupon handles 120,000/month ≈ 30,000/week.

**Scale factor: 12x**

All cost savings and impact estimates will be:
1. Calculated precisely on the sample data
2. Then extrapolated with the 12x factor
3. With explicit uncertainty ranges (conservative/moderate/aggressive)

---

## 4. Deliverable 2: Agentic AI Prototype — Weekly Ops Intelligence Agent

### 4.1 Architecture Overview

The agent uses **LangGraph** for multi-step orchestration with a clear state machine:

```
                    ┌──────────────┐
                    │   START      │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Data Ingest  │  ← Load CSV, validate schema
                    │   & Clean    │  ← Apply all cleaning rules
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Data Quality │  ← Check completeness, anomalies,
                    │   Checks     │     schema violations
                    └──────┬───────┘
                           │
               ┌───────────┼───────────┐
               │           │           │
        ┌──────▼──────┐ ┌──▼────────┐ ┌▼───────────┐
        │   Trend     │ │ Anomaly   │ │   NLP      │  ← Parallel
        │ Detection   │ │ Flagging  │ │ Analysis   │
        └──────┬──────┘ └──┬────────┘ └┬───────────┘
               │           │           │
               └───────────┼───────────┘
                           │
                    ┌──────▼───────┐
                    │ Opportunity  │  ← Synthesize findings,
                    │   Scoring    │     rank by impact
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Recommendation│ ← LLM generates actionable
                    │  Generation   │    recommendations
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Report     │  ← Format as structured
                    │  Generation  │     Markdown brief
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │     END      │
                    └──────────────┘
```

### 4.2 State Schema

```python
from typing import TypedDict, Optional
import pandas as pd

class OpsIntelligenceState(TypedDict):
    # Input
    csv_path: str
    target_week: int  # Which week to analyze (latest by default)
    
    # Data stage
    raw_data: pd.DataFrame
    clean_data: pd.DataFrame
    data_quality_report: dict  # {completeness, anomalies, issues_found}
    
    # Analysis outputs (from parallel nodes)
    trend_analysis: dict      # {metric: {current_week, prior_week, change_pct, direction}}
    anomaly_flags: list       # [{metric, dimension, value, expected_range, severity}]
    nlp_insights: dict        # {sentiment_dist, top_topics, frustration_flags}
    
    # Synthesis
    opportunities: list       # [{title, impact_usd, metric_current, metric_target, root_cause, solution}]
    watch_list: list          # [{pattern, description, severity, trend}]
    
    # Output
    weekly_brief: str         # Final formatted Markdown report
    executive_summary: str    # 3-sentence TL;DR
```

### 4.3 Node Implementations (Detailed)

#### Node 1: `data_ingest_and_clean`
- Load CSV with pandas
- Store raw copy for comparison
- Normalize market labels (map: "United Kingdom"→UK, "GER"→DE, "USA"→US, blank→"Unknown")
- Clamp CSAT to valid range 1-5, set out-of-range to NaN
- Filter negative resolution times to NaN
- Parse timestamps, add derived columns (week_number, day_of_week, hour_of_day)
- Add `is_resolved` boolean column

#### Node 2: `data_quality_checks`
Uses **tool calling** — the LLM calls data profiling tools to inspect the data:

**Tools available to the agent:**
- `check_completeness(df, columns)` → returns missing % per column
- `check_value_ranges(df, column, min_val, max_val)` → returns out-of-range count
- `check_duplicates(df, key_column)` → returns duplicate count
- `detect_statistical_outliers(df, column, method='iqr')` → returns outlier rows

The LLM autonomously decides which checks to run based on the data it sees. This demonstrates **agentic behavior** — the agent inspects the data and decides what quality checks are relevant rather than running a hardcoded list.

#### Node 3: `trend_detection` (Parallel)
Compares target week vs. prior week across key metrics:
- Total ticket volume (overall + by channel, category, team)
- Average FRT, resolution time
- Resolution rate, escalation rate, abandonment rate
- Average CSAT, total and average cost

Output format: `{metric: {current, prior, change_pct, direction, status_emoji}}`

#### Node 4: `anomaly_flagging` (Parallel)
Uses statistical methods:
- **IQR method** for numeric metrics (FRT, resolution time, cost)
- **Z-score method** for rate metrics (resolution %, CSAT)
- **Comparative method** for cross-dimensional analysis (e.g., "BPO VendorB's CSAT for refund tickets this week dropped 0.5 points while other teams stayed flat")

The agent uses tools to run these checks, then uses **LLM reasoning** to determine which anomalies are actionable vs. noise.

#### Node 5: `nlp_analysis` (Parallel)
- **Sentiment analysis**: TextBlob polarity scores on `customer_message`, aggregated by dimension
- **Frustration detection**: Keyword/regex matching for high-frustration signals
- **Topic extraction**: TF-IDF on messages → identify emerging/growing topics vs. prior week
- **Miscategorization candidates**: Use LLM to classify a sample of messages and compare vs. assigned category

#### Node 6: `opportunity_scoring`
The LLM takes all analysis outputs and:
1. Identifies the top 5 opportunities by estimated business impact
2. For each: title, impact sizing (with 12x extrapolation), root cause, proposed solution
3. Creates a "watch list" of 3-5 emerging patterns needing monitoring
4. Uses **structured output** (Pydantic model) for consistent formatting

#### Node 7: `recommendation_generation`
For each opportunity, the LLM generates:
- Specific recommended action (not generic)
- Suggested owner (role/team)
- Priority level (P0/P1/P2)
- Estimated effort and timeline
- Success metric and target

#### Node 8: `report_generation`
Formats everything into a structured Markdown report:

```markdown
# 📊 Ops Intelligence Brief — Week [X]
## [Date Range]

### Executive Summary
[3-sentence TL;DR generated by LLM]

### Key Metrics Dashboard
| Metric | This Week | Last Week | Change | Status |
|--------|-----------|-----------|--------|---------|

### Top 5 Opportunities
#### 1. [Title] — 💰 $XXK/year impact
[Root cause + Solution + Recommended action + owner]

### Watch List
- ⚠️ [Pattern]: [Description]

### Data Quality Notes
[Summary of any data issues]
```

### 4.4 Tool Definitions for Agentic Behavior

```python
@tool
def compute_metric_by_dimension(data, metric, dimension, agg='mean'):
    """Compute a metric grouped by a dimension."""

@tool  
def compare_weeks(data, metric, week_a, week_b):
    """Compare a metric between two weeks."""

@tool
def find_statistical_outliers(data, column, method='iqr'):
    """Find statistical outliers using IQR or Z-score."""

@tool
def analyze_text_sentiment(messages):
    """Compute sentiment scores for messages."""

@tool
def detect_frustration_signals(messages):
    """Detect frustration keywords/patterns."""

@tool
def compute_cost_per_resolved(data, group_by):
    """Compute cost per resolved ticket grouped by a dimension."""

@tool
def run_correlation_analysis(data, target, features):
    """Run correlation analysis between target and features."""
```

### 4.5 Streamlit Frontend

#### Page Layout:
```
┌──────────────────────────────────────────────────────────┐
│  🏢 Groupon Ops Intelligence Command Center              │
│  ─────────────────────────────────────────────────────── │
│  Sidebar:                                                │
│  ├─ 📁 Upload CSV data                                   │
│  ├─ 📅 Select target week                                │
│  ├─ ▶️ Run Analysis button                                │
│  └─ ⚙️ Configuration                                     │
├──────────────────────────────────────────────────────────┤
│  Main Area (tabs):                                       │
│  ┌─────────┬──────────┬──────────┬──────────┬──────────┐ │
│  │Dashboard│Opportuni-│ Trends  │ NLP     │ Weekly   │ │
│  │ Overview│ties      │         │ Insights│ Brief    │ │
│  └─────────┴──────────┴──────────┴──────────┴──────────┘ │
│                                                          │
│  Tab 1: Key metric cards + trend charts                  │
│  Tab 2: Ranked opportunity cards                         │
│  Tab 3: Week-over-week comparison charts                 │
│  Tab 4: Sentiment, topics, frustration signals           │
│  Tab 5: Full Markdown report                             │
├──────────────────────────────────────────────────────────┤
│  Bottom: Agent activity log (multi-step reasoning)       │
└──────────────────────────────────────────────────────────┘
```

#### Visualizations Plan:
1. **KPI Cards Row**: Total tickets, Avg FRT, Resolution Rate, Avg CSAT, Total Cost — each with week-over-week delta
2. **Ticket Volume by Channel/Category**: Stacked bar chart (Plotly)
3. **Team Performance Radar**: Radar chart comparing teams on FRT, resolution rate, CSAT, cost
4. **FRT Distribution**: Box plots by team, overlaid with target SLA lines
5. **CSAT Heatmap**: Matrix of CSAT by category × team (Plotly heatmap)
6. **Trend Lines**: Multi-line chart of key metrics over weeks
7. **Anomaly Highlights**: Annotated scatter plots showing flagged data points
8. **NLP Word Cloud / Topic Chart**: Top topics with sentiment overlay

### 4.6 Demo Flow (Screen Recording Script, ~4 minutes)

1. **Open Streamlit app** → show the clean UI (10s)
2. **Upload CSV** or show pre-loaded (10s)
3. **Click "Run Analysis"** → show agent pipeline executing step by step (30s)
4. **Dashboard tab** → walk through KPI cards, highlight deltas (30s)
5. **Opportunities tab** → show top 5 ranked, click into one for root cause (45s)
6. **Trends tab** → show week-over-week charts, point out anomaly (30s)
7. **NLP Insights tab** → sentiment distribution, frustration signals (30s)
8. **Weekly Brief tab** → show full Markdown report, download it (20s)
9. **Change target week** → show agent re-runs with different insights (20s)
10. **Closing** → recap the value proposition (15s)

---

## 5. Deliverable 3: Scaling Roadmap

### Slide 1: From Prototype to Production

```
Phase 1 (Month 1-2): Production Pipeline
├── Deploy on scheduled cron (every Monday 6am)
├── Connect to Zendesk/Salesforce API for live ticket data
├── Slack integration for automated brief delivery
└── Alerting for critical anomalies (PagerDuty/email)

Phase 2 (Month 3-4): Enhanced Intelligence
├── Real-time streaming (detect issues intra-week)
├── Asana/Jira integration for action item tracking
├── Add NPS + revenue data for impact correlation
└── Multi-language NLP for non-English markets

Phase 3 (Month 5-6): Autonomous Operations
├── Predictive models: forecast ticket spikes
├── Auto-routing: AI decides optimal team assignment
├── Self-healing: auto-trigger workforce scaling
└── Feedback loop: track recommendation implementation + impact
```

### Slide 2: Data Sources & Governance

**Additional Data Sources:**
- CRM data (customer LTV, purchase history)
- Workforce management (agent schedules, utilization)
- Revenue data (deal performance, market-level GMV)
- Product telemetry (app crashes, broken checkout → predict ticket spikes)

**Governance & Human-in-the-Loop:**
- All AI recommendations surfaced as suggestions, not auto-executed (initially)
- Weekly review meeting where SVP signs off on top actions
- Decision audit trail: every recommendation tracked for implementation and outcome
- Bias monitoring: ensure AI doesn't systematically deprioritize certain markets
- Data privacy: PII redaction before NLP analysis, GDPR compliance for EU markets

---

## 6. Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| **Language** | Python 3.11+ | Standard for data science + AI |
| **Agent Framework** | LangGraph | Best-in-class for multi-step stateful workflows |
| **LLM** | Claude (Anthropic) via `langchain-anthropic` | Strong reasoning, tool use, structured output |
| **Data Processing** | pandas, numpy | Standard, reliable |
| **NLP** | TextBlob (sentiment), scikit-learn (TF-IDF, clustering) | Lightweight, no heavy model dependencies |
| **Visualization** | Plotly Express | Interactive charts that work in Streamlit |
| **Frontend** | Streamlit | Fast to build, polished result, great for demos |
| **Code Quality** | ruff (linting) | Professional code standards |
| **Environment** | Python venv + pip | Simple, reliable |

### Dependencies (requirements.txt)
```
langchain>=0.3.0
langchain-anthropic>=0.3.0
langchain-core>=0.3.0
langgraph>=0.3.0
streamlit>=1.40.0
pandas>=2.2.0
numpy>=1.26.0
plotly>=5.24.0
textblob>=0.18.0
scikit-learn>=1.5.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

---

## 7. Project Structure

```
G/
├── .env                          # API keys (ANTHROPIC_API_KEY)
├── .env.example                  # Template for .env
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
│   ├── config.py                  # Configuration constants
│   ├── data_cleaning.py           # Data loading and cleaning
│   ├── analytics.py               # Statistical analysis
│   ├── nlp_analysis.py            # NLP: sentiment, topics, frustration
│   ├── visualizations.py          # Plotly chart functions
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py               # LangGraph state definition
│   │   ├── tools.py               # Tool definitions
│   │   ├── nodes.py               # Node functions
│   │   ├── graph.py               # LangGraph graph assembly
│   │   └── prompts.py             # System prompts
│   │
│   └── app/
│       ├── streamlit_app.py       # Main Streamlit app
│       ├── components.py          # Reusable UI components
│       └── styles.py              # Custom CSS/theming
│
├── notebooks/
│   └── eda_analysis.ipynb         # EDA notebook
│
├── output/                        # Generated weekly briefs
│
└── tests/
    ├── test_data_cleaning.py
    ├── test_analytics.py
    └── test_agent.py
```

---

## 8. Presentation & Delivery Strategy

### 8.1 Slide Deck Structure (Max 15 pages)

| Slide # | Content | Purpose |
|---|---|---|
| 1 | Title: "AI-Powered Ops Intelligence: From Data to Decisions" | First impression |
| 2 | The Problem: 120K tickets/month, no unified view | Set the stage |
| 3 | My Approach: methodology overview | Show structured thinking |
| 4 | Data Quality: what I found, how I handled it | Demonstrate rigor |
| 5 | EDA: Key metrics dashboard | Analytical depth |
| 6 | Deep Dive: Team performance comparison | Show the data work |
| 7 | Opportunity #1 card (biggest prize) | Impact-first storytelling |
| 8 | Opportunity #2 card | |
| 9 | Opportunity #3 card | |
| 10 | Opportunities #4 & #5 (condensed) | |
| 11 | Impact Summary: effort/impact matrix | Strategic clarity |
| 12 | The Agent: architecture + what it does | Technical credibility |
| 13 | Demo screenshots / key outputs | For async reviewers |
| 14 | Scaling Roadmap | Strategic vision |
| 15 | Summary + first 90 days as CoS | Close with confidence |

### 8.2 Screen Recording (Loom, ~4-5 minutes)

- Start with brief context (10 seconds)
- Walk through the live demo (3.5 minutes)
- End with "here's what I'd build next" (30 seconds)
- Professional but natural — not scripted word-for-word

---

## 9. Risk Mitigation

| Risk | Mitigation |
|---|---|
| LLM API costs during development | Use claude-haiku for dev, sonnet for demo |
| LLM generates inconsistent output | Pydantic structured output throughout; validate with schemas |
| Streamlit performance with 10K rows | Pre-compute all analytics, cache with @st.cache_data |
| Agent takes too long to run | Parallelize independent analysis nodes in LangGraph |
| Demo breaks during recording | Dry run first; have fallback static screenshots |
| API key exposure | .env + .gitignore, use .env.example for documentation |

---

## 10. Timeline

| Day | Focus | Deliverables |
|---|---|---|
| **Day 1** | Data pipeline + EDA | `data_cleaning.py`, `analytics.py`, EDA notebook |
| **Day 2** | NLP analysis + opportunity sizing | `nlp_analysis.py`, final top 5 quantified |
| **Day 3** | Agent pipeline (LangGraph) | `agent/` package fully functional |
| **Day 4** | Streamlit app + visualizations | `streamlit_app.py` with all tabs |
| **Day 5** | Slide deck + recording + polish | Slides PDF, Loom recording, final cleanup |

---

## 11. Progress Tracker

### Phase 0: Planning & Setup
- [x] Data profiling and initial analysis complete
- [x] Implementation plan written
- [x] Agent instructions file created (`AGENT_INSTRUCTIONS.md`)
- [x] Project scaffolding set up (30 files, full directory structure)
- [x] GitHub repository initialized (local commit `c3d0731`)
- [ ] GitHub remote created and pushed (private repo — pending auth)

### Phase 1: Core Modules (Scaffolded — Ready for Implementation)
- [x] `src/config.py` — Constants, branding, frustration patterns
- [x] `src/data_cleaning.py` — Load, clean, quality report functions
- [x] `src/analytics.py` — KPI summary, team/channel/category perf, outliers, correlations
- [x] `src/nlp_analysis.py` — Sentiment, frustration detection, TF-IDF topic extraction
- [x] `src/visualizations.py` — 12 Plotly chart functions + KPI card HTML

### Phase 2: Agent Pipeline (Scaffolded — Ready for Implementation)
- [x] `src/agent/state.py` — AgentState TypedDict, OpportunityItem, WeeklyBrief
- [x] `src/agent/tools.py` — Tool wrappers for analytics & NLP functions
- [x] `src/agent/nodes.py` — 6 pipeline nodes (ingest → quality → analysis → scoring → report)
- [x] `src/agent/graph.py` — LangGraph StateGraph with fan-out/fan-in topology
- [x] `src/agent/prompts.py` — Opportunity scoring, report generation, trend interpretation prompts

### Phase 3: Frontend (Scaffolded — Ready for Implementation)
- [x] `src/app/streamlit_app.py` — 5-tab app with caching and Groupon branding
- [x] `src/app/components.py` — Tab render functions (Dashboard, Opportunities, Trends, NLP, Brief)
- [x] `src/app/styles.py` — Custom CSS injection for Groupon green theme

### Phase 4: Testing & Quality
- [x] `tests/test_data_cleaning.py` — 5 tests (load, columns, market normalization, CSAT, resolution)
- [x] `tests/test_analytics.py` — KPI, team perf, chatbot escalation tests
- [x] `tests/test_nlp.py` — Sentiment and frustration detection tests
- [ ] All tests passing with real data
- [ ] Code reviewed and cleaned

### Phase 5: Deliverables
- [ ] EDA analysis complete (Jupyter notebook)
- [ ] Top 5 opportunities quantified with dollar impact
- [ ] Slide deck created (15 pages max)
- [ ] Screen recording done (~4 min demo)
- [ ] Final submission prepared