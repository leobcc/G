# AI-Powered Customer Operations Command Center

An AI-powered Operations Intelligence system for Groupon's Customer Operations. Automatically analyzes support ticket data and produces weekly intelligence briefs with actionable improvement opportunities.

## Overview

This system ingests customer support ticket data (~120,000 tickets/month), runs a multi-step analytical pipeline using an agentic AI architecture (LangGraph), and produces a structured weekly "Ops Intelligence Brief" identifying the top 5 improvement opportunities — each quantified by business impact and paired with actionable recommendations.

## Features

- **Automated Data Cleaning**: Handles messy, real-world data with missing values, inconsistencies, and edge cases
- **Multi-Step Agent Pipeline**: LangGraph-powered agent with 7 analysis stages running in parallel where possible
- **NLP Analysis**: Sentiment scoring, frustration detection, and topic extraction from customer messages
- **Interactive Dashboard**: Streamlit-based UI with Plotly charts for exploring insights
- **Weekly Intelligence Brief**: Structured Markdown report with KPIs, opportunities, and watch list
- **Week-over-Week Comparison**: Trend detection and anomaly flagging

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ (tested on 3.13) |
| Agent Framework | LangGraph |
| LLM | Google Gemini (free tier) |
| Data Processing | pandas, numpy |
| NLP | TextBlob, scikit-learn |
| Visualization | Plotly Express |
| Frontend | Streamlit |

## Key Findings (from ~10,000 ticket sample)

| Opportunity | Annual Impact | Effort | Timeline |
|---|---|---|---|
| Channel-Mix Optimization (Email → Chat) | $215,731 | Low | 4 weeks |
| Reduce Abandoned Ticket Rate (8.3% → 4%) | $198,985 | Low | 3 weeks |
| Reduce Chatbot Escalation (25.7% → 15%) | $192,408 | Medium | 8 weeks |
| BPO Vendor B Performance Gap | $148,679 | Medium | 6 weeks |
| CSAT Recovery Program | $1,382,400 | Medium | 6 weeks |
| **Total** | **$2,138,203** | | |

*Extrapolated from 10K sample using 12x scale factor to Groupon's ~120K tickets/month*

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd G
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY (free Gemini API key)
# Get one at: https://aistudio.google.com/apikey
```

### 3. Run the dashboard

```bash
# Windows (PowerShell)
$env:PYTHONPATH = (Get-Location).Path
streamlit run src/app/streamlit_app.py --server.headless true

# macOS/Linux
PYTHONPATH=$(pwd) streamlit run src/app/streamlit_app.py
```

### 4. Run tests

```bash
# Windows (PowerShell)
$env:PYTHONPATH = (Get-Location).Path
pytest tests/ -v

# macOS/Linux
PYTHONPATH=$(pwd) pytest tests/ -v
```

### 5. Explore the EDA notebook

Open `notebooks/eda_analysis.ipynb` in VS Code or Jupyter for the full exploratory analysis with interactive Plotly charts.

## Project Structure

```
├── data/                    # Raw ticket data (10K tickets)
├── src/
│   ├── config.py            # Configuration constants
│   ├── data_cleaning.py     # Data loading and cleaning
│   ├── analytics.py         # Statistical analysis functions
│   ├── nlp_analysis.py      # NLP: sentiment, frustration, topics
│   ├── visualizations.py    # Plotly chart functions
│   ├── agent/               # LangGraph agent pipeline
│   │   ├── state.py         # TypedDict state definition
│   │   ├── tools.py         # Agent tools (pure Python)
│   │   ├── nodes.py         # Pipeline nodes (7 stages)
│   │   ├── graph.py         # Graph assembly with fan-out/fan-in
│   │   └── prompts.py       # LLM system prompts
│   └── app/                 # Streamlit application
│       ├── streamlit_app.py # Main 5-tab dashboard
│       ├── components.py    # Reusable UI components
│       └── styles.py        # Custom CSS (Groupon branding)
├── notebooks/
│   └── eda_analysis.ipynb   # Full EDA with 22 sections
├── output/                  # Generated reports and charts
└── tests/                   # Test suite (25 tests)
```

## Data Notes

- **Source**: `data/option_a_ticket_data.csv` — 10,000 tickets, 16 columns
- **Date range**: Feb 9 – Mar 10, 2026 (Weeks 7–11)
- **Week 11**: Partial (224 tickets only) — excluded from week-over-week comparisons
- **Scale factor**: 12x (sample → estimated Groupon monthly volume of 120K tickets)
- **Cleaning**: 42 market labels normalized, 53 CSAT scores clamped, 71 negative resolution times fixed; no rows dropped

## Business Case

This project was built as part of a business case assignment for the **Chief of Staff - Global Operations (AI-First)** position at Groupon Madrid. It demonstrates:

1. **Analytical depth**: Cleaning messy data, extracting signal, sizing opportunities
2. **AI-first thinking**: Agent-based architecture, not just dashboards
3. **Technical execution**: Working prototype, not just slides
4. **Strategic clarity**: Connecting operational metrics to business impact
5. **Communication**: Executive-ready outputs and clear presentation
