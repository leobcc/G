**BUSINESS CASE ASSIGNMENT**

Chief of Staff - Global Operations (AI-First)

Groupon - Madrid

| **Candidate**   | \[Candidate Name\]                                                           |
| --------------- | ---------------------------------------------------------------------------- |
| **Date Issued** | \[Date\]                                                                     |
| ---             | ---                                                                          |
| **Deadline**    | 5 working days from receipt                                                  |
| ---             | ---                                                                          |
| **Format**      | Slide deck or document (max 15 pages) + screen recording or live demo link   |
| ---             | ---                                                                          |
| **Delivery**    | Slide deck / document + screen recording of working demo (or live demo link) |
| ---             | ---                                                                          |

**Purpose**

This assignment evaluates your ability to operate as a strategic, AI-first Chief of Staff in a real-world operations environment. We are not looking for polished consulting decks or generic AI hype. We want to see how you think, how you use data, and critically, how you build and deploy AI-powered solutions that go beyond prompting-toward agentic, automated, and production-grade workflows.

**What We Are Evaluating**

- Analytical depth: Can you work with messy data, extract signal, and size opportunities?
- AI-first thinking: Do you default to AI/automation when designing solutions, not as an afterthought?
- Technical execution: Can you build something that works-not just describe it?
- Strategic clarity: Can you connect operational detail to business impact with clear logic?
- Communication: Can you present a complex analysis simply and persuasively?

**Important Notes**

- You may use any tools, platforms, or AI assistants. In fact, we expect you to. Show us your toolkit and workflow.
- We value a working prototype over a perfect deck. Ship something real.
- **If you make assumptions, state them explicitly. Smart assumptions are a strength. When something is unclear, make an assumption**
- The synthetic data provided is intentionally imperfect. That is the point.

**OPTION A**

**AI-Powered Customer Operations Command Center**

**Scenario**

Groupon's Customer Operations team handles approximately 120,000 support tickets per month across multiple channels (email, chat, phone, social). The operation is currently managed through a mix of in-house agents, BPO partners, and a basic AI chatbot. Leadership suspects there are significant inefficiencies in ticket routing, resolution times, and agent utilization-but lacks a unified, data-driven view of where the biggest opportunities lie.

You have just joined as Chief of Staff. The SVP of Global Operations asks you to build an "Operations Intelligence" system that can automatically identify the top 3-5 improvement opportunities each week, backed by data and ready for action.

**Synthetic Data Provided**

You will receive a CSV dataset (~10,000 rows) simulating 4 weeks of ticket data with the following fields:

| **Field**           | **Description**                                                                |
| ------------------- | ------------------------------------------------------------------------------ |
| ticket_id           | Unique identifier                                                              |
| ---                 | ---                                                                            |
| created_at          | Timestamp (UTC)                                                                |
| ---                 | ---                                                                            |
| channel             | email / chat / phone / social                                                  |
| ---                 | ---                                                                            |
| category            | refund, order_status, merchant_issue, billing, account, voucher_problem, other |
| ---                 | ---                                                                            |
| subcategory         | More granular issue type (some missing)                                        |
| ---                 | ---                                                                            |
| priority            | low / medium / high / urgent                                                   |
| ---                 | ---                                                                            |
| customer_message    | Free-text initial customer message                                             |
| ---                 | ---                                                                            |
| assigned_team       | in_house / bpo_vendorA / bpo_vendorB / ai_chatbot                              |
| ---                 | ---                                                                            |
| agent_id            | Agent identifier (null for chatbot)                                            |
| ---                 | ---                                                                            |
| first_response_min  | Minutes to first response                                                      |
| ---                 | ---                                                                            |
| resolution_min      | Minutes to resolution (null if unresolved)                                     |
| ---                 | ---                                                                            |
| resolution_status   | resolved / escalated / abandoned / pending                                     |
| ---                 | ---                                                                            |
| csat_score          | 1-5 (null if not collected)                                                    |
| ---                 | ---                                                                            |
| contacts_per_ticket | Number of back-and-forth exchanges                                             |
| ---                 | ---                                                                            |
| cost_usd            | Estimated cost per ticket                                                      |
| ---                 | ---                                                                            |
| market              | US / UK / DE / FR / ES / IT / AU                                               |
| ---                 | ---                                                                            |

_Note: The data contains intentional noise-missing values, inconsistencies, and edge cases. Data quality handling is part of the evaluation._

**Your Deliverables**

**1\. Exploratory Analysis & Opportunity Sizing (Slides)**

- Clean and analyze the dataset. Identify the top 5 operational improvement opportunities ranked by estimated business impact (cost savings, CSAT improvement, or efficiency gains).
- For each opportunity: quantify the prize, explain the root cause, and propose an AI-first solution.
- Show your analytical work-not just conclusions. We want to see how you explored the data.

**2\. Agentic AI Prototype: Weekly Ops Intelligence Agent (Live Demo)**

Build a working AI agent (not a simple prompt/chat) that:

- Ingests the ticket CSV data automatically
- Runs a multi-step analytical pipeline: data quality checks → trend detection → anomaly flagging → opportunity identification → recommendation generation
- Produces a structured weekly "Ops Intelligence Brief" (e.g., Markdown report, dashboard, or Slack-formatted output) with: top 5 issues this week, comparison vs. prior week, specific recommended actions with owners, and a "watch list" of emerging patterns
- Demonstrates agent-like behavior: tool use, multi-step reasoning, memory/context across steps, or autonomous decision-making (e.g., deciding which analyses to run based on what the data shows)

_Acceptable tools/platforms (non-exhaustive): Python + LangChain/LangGraph/CrewAI, Claude/OpenAI function calling with tool use, n8n/Make with AI nodes, Streamlit/Gradio front-end, or any framework that shows agentic capability._

**3\. Scaling Roadmap (1-2 slides)**

- How would you evolve this prototype into a production system that runs autonomously every Monday morning and distributes insights to the right stakeholders?
- What data sources would you add? What integrations (Zendesk, Slack, Asana, BI tools)?
- What governance and human-in-the-loop controls would you design?

**OPTION B**

**Merchant Operations AI Transformation - From Analysis to Autonomous Action**

**Scenario**

Groupon's Merchant Operations team manages the lifecycle of deals on the marketplace: onboarding new merchants, managing deal quality, handling merchant escalations, and monitoring deal performance. The team of ~40 FTEs (mix of in-house and BPO) processes approximately 3,500 merchant requests per week across deal setup, content changes, performance complaints, and billing disputes.

The SVP of Global Operations believes this function is ripe for AI-driven transformation, but needs a data-backed assessment of where to start and a proof-of-concept showing what "AI-first merchant ops" could actually look like. Your job is to be that person.

**Synthetic Data Provided**

You will receive two linked CSV files:

**File 1: merchant_requests.csv (~8,000 rows, 4 weeks)**

| **Field**           | **Description**                                                                                     |
| ------------------- | --------------------------------------------------------------------------------------------------- |
| request_id          | Unique identifier                                                                                   |
| ---                 | ---                                                                                                 |
| created_at          | Timestamp (UTC)                                                                                     |
| ---                 | ---                                                                                                 |
| merchant_id         | Merchant identifier                                                                                 |
| ---                 | ---                                                                                                 |
| merchant_tier       | platinum / gold / silver / bronze (based on GMV)                                                    |
| ---                 | ---                                                                                                 |
| request_type        | deal_setup / content_change / performance_complaint / billing_dispute / account_update / escalation |
| ---                 | ---                                                                                                 |
| request_description | Free-text description from merchant or internal team                                                |
| ---                 | ---                                                                                                 |
| assigned_to         | in_house / bpo_team / unassigned                                                                    |
| ---                 | ---                                                                                                 |
| sla_hours           | Target SLA for this request type and tier                                                           |
| ---                 | ---                                                                                                 |
| actual_hours        | Actual time to completion (null if open)                                                            |
| ---                 | ---                                                                                                 |
| sla_met             | true / false / null                                                                                 |
| ---                 | ---                                                                                                 |
| resolution          | completed / rejected / escalated / pending / auto_resolved                                          |
| ---                 | ---                                                                                                 |
| touches             | Number of human interactions before resolution                                                      |
| ---                 | ---                                                                                                 |
| cost_usd            | Estimated processing cost                                                                           |
| ---                 | ---                                                                                                 |
| market              | US / UK / DE / FR / ES / IT / AU                                                                    |
| ---                 | ---                                                                                                 |

**File 2: merchant_profiles.csv (~1,200 rows)**

| **Field**        | **Description**                                                     |
| ---------------- | ------------------------------------------------------------------- |
| merchant_id      | Links to requests table                                             |
| ---              | ---                                                                 |
| merchant_name    | Business name                                                       |
| ---              | ---                                                                 |
| category         | health_beauty / food_drink / activities / travel / goods / services |
| ---              | ---                                                                 |
| active_deals     | Number of live deals                                                |
| ---              | ---                                                                 |
| monthly_gmv_usd  | Gross merchandise value (last 30 days)                              |
| ---              | ---                                                                 |
| lifetime_months  | Months on Groupon platform                                          |
| ---              | ---                                                                 |
| churn_risk_score | 0-1 probability (from existing model, noisy)                        |
| ---              | ---                                                                 |
| nps_last_survey  | −10 to 10 (null if never surveyed)                                  |
| ---              | ---                                                                 |

_Note: As with real data, expect noise, missing values, and some inconsistencies. How you handle this is part of the evaluation._

**Your Deliverables**

**1\. Merchant Ops Diagnostic & AI Transformation Roadmap (Slides)**

- Analyze both datasets to build a full picture of Merchant Ops performance: where are the bottlenecks, cost drivers, SLA failures, and highest-value merchant pain points?
- Segment the analysis by merchant tier, request type, and team to identify where AI could have the most impact.
- Propose a phased AI transformation plan (3/6/12 months) with estimated impact for each phase.
- Show the analytical work. We want to see your EDA process, not just a summary of findings.

**2\. Agentic AI Prototype: Intelligent Request Triage & Action Agent (Live Demo)**

Build a working AI agent that demonstrates what "AI-first merchant ops" could look like. The agent should:

- Ingest a new merchant request (free-text description + structured metadata) and automatically: classify the request type and urgency, enrich it with merchant context (tier, GMV, churn risk, history), decide the optimal routing (auto-resolve, assign to specialist, escalate to manager, or flag for review), and for auto-resolvable requests, draft the response.
- Handle multi-step reasoning: e.g., if a platinum merchant with high churn risk submits a billing dispute, the agent should recognize this is high-priority, pull merchant history, flag the churn risk to the account manager, and draft a personalized response-not just route it.
- Include a "batch mode" that can process a queue of 20-50 pending requests and produce a prioritized work queue with AI-recommended actions for each.

_The agent must go beyond prompt-and-response. We are looking for: tool use (calling functions, querying data), multi-step orchestration (agent decides what to do next based on intermediate results), structured output (not just free text-actionable routing decisions, priority scores, drafted responses), and error handling / fallback logic._

**3\. Business Impact Model (1 slide)**

- Based on your analysis and prototype, estimate: what percentage of requests could be auto-resolved or auto-triaged? What is the projected FTE reduction or cost savings? What is the expected impact on SLA compliance and merchant satisfaction?

**Evaluation Rubric**

**CHOOSING YOUR OPTION**

|                     | **Option A: Customer Ops Command Center**                       | **Option B: Merchant Ops AI Transformation**                                   |
| ------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **Focus**           | Customer support analytics + monitoring agent                   | Merchant operations + intelligent triage agent                                 |
| ---                 | ---                                                             | ---                                                                            |
| **Data Complexity** | Single dataset, more NLP-heavy (customer messages)              | Two linked datasets, more relational/business logic                            |
| ---                 | ---                                                             | ---                                                                            |
| **Agent Type**      | Analytical/monitoring agent (runs on schedule)                  | Operational/decision agent (acts on individual requests)                       |
| ---                 | ---                                                             | ---                                                                            |
| **Best For**        | Candidates stronger in analytics, NLP, and reporting automation | Candidates stronger in workflow design, business logic, and process automation |
| ---                 | ---                                                             | ---                                                                            |

_Choose the option that best showcases your strengths. Both are equally weighted in our evaluation._

**Logistics**

- Deadline: 5 working days from the date this assignment is received.
- Submit your slide deck or document (PDF / Google Slides / Google Docs link), a link to your code repository (GitHub, GitLab, or similar), and a screen recording of your demo (Loom or similar) or a live demo link.

**Good luck. Show us how you think, how you build, and how you would change how we operate.**