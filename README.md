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
| Language | Python 3.11+ |
| Agent Framework | LangGraph |
| LLM | Claude (Anthropic) |
| Data Processing | pandas, numpy |
| NLP | TextBlob, scikit-learn |
| Visualization | Plotly Express |
| Frontend | Streamlit |

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd G
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run the app

```bash
streamlit run src/app/streamlit_app.py
```

### 4. Run tests

```bash
python -m pytest tests/
```

## Project Structure

```
├── data/                    # Raw ticket data
├── src/
│   ├── config.py            # Configuration constants
│   ├── data_cleaning.py     # Data loading and cleaning
│   ├── analytics.py         # Statistical analysis
│   ├── nlp_analysis.py      # NLP: sentiment, topics
│   ├── visualizations.py    # Plotly chart functions
│   ├── agent/               # LangGraph agent pipeline
│   │   ├── state.py         # State definition
│   │   ├── tools.py         # Agent tools
│   │   ├── nodes.py         # Pipeline nodes
│   │   ├── graph.py         # Graph assembly
│   │   └── prompts.py       # LLM prompts
│   └── app/                 # Streamlit application
│       ├── streamlit_app.py
│       ├── components.py
│       └── styles.py
├── notebooks/               # EDA notebooks
├── output/                  # Generated reports
└── tests/                   # Test suite
```

## Business Case

This project was built as part of a business case assignment for the **Chief of Staff - Global Operations (AI-First)** position at Groupon Madrid. It demonstrates:

1. **Analytical depth**: Cleaning messy data, extracting signal, sizing opportunities
2. **AI-first thinking**: Agent-based architecture, not just dashboards
3. **Technical execution**: Working prototype, not just slides
4. **Strategic clarity**: Connecting operational metrics to business impact
5. **Communication**: Executive-ready outputs and clear presentation
