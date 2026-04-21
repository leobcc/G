# AI-Powered Customer Operations Command Center

This repository analyzes customer support ticket data, runs an agentic weekly ops pipeline, and generates a structured **Ops Intelligence Brief** with the top improvement opportunities, trend changes, and recommended actions.

## What this repo includes

- **Exploratory analysis** in [notebooks/eda_analysis.ipynb](notebooks/eda_analysis.ipynb)
- **Interactive Streamlit app** in [src/app/streamlit_app.py](src/app/streamlit_app.py)
- **Agentic workflow** in [src/agent](src/agent)
- **Deterministic analytics + NLP** in [src](src)
- **Tests** in [tests](tests)
- **Slide / roadmap content** in [slides](slides)

## Main capabilities

- Cleans messy support ticket data with explicit handling of invalid and missing values
- Computes KPI summaries, weekly trends, team/channel/category performance, and outliers
- Analyzes customer text for sentiment, frustration, and topic signals
- Runs a LangGraph pipeline to synthesize findings into opportunities and a weekly brief
- Presents results in a 5-tab Streamlit dashboard

## App overview

The app includes five views:

- **Dashboard** — KPIs, performance comparisons, and trends
- **NLP Insights** — sentiment, frustration, and text patterns
- **Trends** — week-over-week movements and anomaly review
- **Opportunities** — prioritized improvement actions with impact estimates
- **Weekly Brief** — a generated markdown summary for leadership review

## Architecture

The project has two main layers:

### 1. Deterministic analytics
Core modules:
- [src/data_cleaning.py](src/data_cleaning.py)
- [src/analytics.py](src/analytics.py)
- [src/nlp_analysis.py](src/nlp_analysis.py)
- [src/visualizations.py](src/visualizations.py)

### 2. Agentic synthesis
Core modules:
- [src/agent/graph.py](src/agent/graph.py)
- [src/agent/nodes.py](src/agent/nodes.py)
- [src/agent/tools.py](src/agent/tools.py)
- [src/agent/prompts.py](src/agent/prompts.py)

Pipeline flow:

**ingest → data quality → parallel analysis (trends / anomalies / NLP) → opportunity scoring → report generation → executive insights**

## Tech stack

| Component | Implementation |
|---|---|
| Language | Python 3.11+ |
| Frontend | Streamlit |
| Agent Framework | LangGraph + LangChain |
| LLM | Groq (`meta-llama/llama-4-scout-17b-16e-instruct`) |
| Data Processing | pandas, numpy |
| Statistics / ML | scipy, scikit-learn |
| NLP | vaderSentiment + sklearn-based text workflows |
| Visualization | Plotly |
| Testing | pytest |

## Key findings from the sample dataset

Top modeled opportunities identified in the analysis:

| Rank | Opportunity | Estimated Annual Impact |
|---|---|---:|
| 1 | Email to chat deflection | $215,731 |
| 2 | Abandonment reduction | $199,861 |
| 3 | Chatbot escalation reduction | $192,408 |
| 4 | Proactive CSAT recovery | $184,176 |
| 5 | Vendor B quality improvement | $49,561 |
|  | **Total modeled impact** | **$841,737** |

A few representative findings:

- Email is the largest and one of the least efficient channels
- Refund and order-status tickets drive a large share of workload
- Vendor B materially underperforms on speed and CSAT
- The chatbot is extremely cheap and fast, but escalates too often in the wrong categories
- Volume rises across the complete weeks without clear efficiency gains

## Repository structure

```text
G/
├── data/                       # Input dataset
├── docs/                       # Assignment materials
├── notebooks/                  # EDA notebook
├── output/                     # Generated artifacts
├── slides/                     # Slide content and roadmap notes
├── src/
│   ├── analytics.py
│   ├── config.py
│   ├── data_cleaning.py
│   ├── nlp_analysis.py
│   ├── visualizations.py
│   ├── agent/
│   └── app/
├── tests/
├── requirements.txt
└── README.md
```

## Local setup

### 1. Create a virtual environment

```bash
git clone <repo-url>
cd G
python -m venv .venv
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy [.env.example](.env.example) to `.env` and add your Groq API key:

```dotenv
GROQ_API_KEY=your-groq-api-key-here
```

### 4. Run the app

```bash
streamlit run src/app/streamlit_app.py
```

### 5. Run tests

```bash
pytest tests/ --tb=short -q
```

## Data and methodology notes

- Source data: [data/option_a_ticket_data.csv](data/option_a_ticket_data.csv)
- Comparable weeks: **7–10**
- Week 11 is partial and excluded from week-over-week interpretation
- Raw data is not modified in place
- Opportunity sizing uses explicit scale assumptions from the project config and analysis

## Notes for reviewers

If reviewing quickly, the best path is:

1. Open the app in [src/app/streamlit_app.py](src/app/streamlit_app.py)
2. Review the EDA in [notebooks/eda_analysis.ipynb](notebooks/eda_analysis.ipynb)
3. Inspect the agent flow in [src/agent/graph.py](src/agent/graph.py)
4. Check the tests in [tests](tests)

## Limitations

- The dataset is synthetic and intentionally noisy
- Annualized impact depends on stated scale assumptions
- LLM-backed stages depend on API availability and rate limits
- The current version is a prototype, not a production deployment
