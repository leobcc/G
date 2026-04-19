"""Smoke test: verify full pipeline runs on original data without errors."""
import sys
from pathlib import Path

# Add project root so `src` is importable as a package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_cleaning import load_raw_data, clean_data, detect_complete_weeks
from src.analytics import (
    compute_kpi_summary,
    compute_weekly_trends,
    compute_wow_kpis,
    compute_team_performance,
    compute_channel_performance,
    compute_category_performance,
    compute_chatbot_escalation_analysis,
)

raw = load_raw_data("data/option_a_ticket_data.csv")
clean_df, log = clean_data(raw)
cw = detect_complete_weeks(clean_df)
print(f"Rows: {len(clean_df)}, Weeks detected: {cw}")

kpis = compute_kpi_summary(clean_df)
print(f"KPIs: total_tickets={kpis['total_tickets']}")

trends = compute_weekly_trends(clean_df, complete_weeks=cw)
print(f"Trends shape: {trends.shape}")

wow = compute_wow_kpis(clean_df, complete_weeks=cw)
print(f"WoW current_week: {wow['current_week']}")

tp = compute_team_performance(clean_df)
print(f"Teams: {len(tp)} rows")

cp = compute_channel_performance(clean_df)
print(f"Channels: {len(cp)} rows")

cat = compute_category_performance(clean_df)
print(f"Categories: {len(cat)} rows")

bot = compute_chatbot_escalation_analysis(clean_df)
print(f"Chatbot esc rate: {bot['overall_escalation_rate']:.3f}")

print("\n=== ALL OK ===")

# Now test with synthetic CSV (different weeks)
print("\n--- Testing with synthetic data (different weeks) ---")
import numpy as np
import pandas as pd
from datetime import timedelta

rng = np.random.default_rng(123)
n = 300
start = pd.Timestamp("2025-03-03")
dates = [start + timedelta(hours=int(rng.integers(0, 21 * 24))) for _ in range(n)]

synth = pd.DataFrame({
    "ticket_id": [f"S-{i}" for i in range(n)],
    "created_at": dates,
    "channel": rng.choice(["email", "chat", "phone"], size=n),
    "category": rng.choice(["refund", "billing", "order_status"], size=n),
    "subcategory": rng.choice(["sub_a", "sub_b"], size=n),
    "priority": rng.choice(["low", "medium", "high"], size=n),
    "assigned_team": rng.choice(["in_house", "bpo_vendorA", "ai_chatbot"], size=n),
    "resolution_status": rng.choice(["resolved", "escalated", "abandoned"], size=n, p=[0.6, 0.25, 0.15]),
    "first_response_min": rng.exponential(25, size=n).round(1),
    "resolution_min": rng.exponential(50, size=n).round(1),
    "csat_score": rng.uniform(1, 5, size=n).round(1),
    "customer_message": [f"Msg {i}" for i in range(n)],
    "cost_usd": rng.uniform(2, 8, size=n).round(2),
    "market": rng.choice(["US", "UK"], size=n),
    "customer_contacts": rng.integers(1, 4, size=n),
})

clean_synth, log_synth = clean_data(synth)
cw_synth = detect_complete_weeks(clean_synth)
print(f"Synth rows: {len(clean_synth)}, Weeks detected: {cw_synth}")

kpis_synth = compute_kpi_summary(clean_synth)
print(f"KPIs: total={kpis_synth['total_tickets']}")

trends_synth = compute_weekly_trends(clean_synth, complete_weeks=cw_synth)
print(f"Trends shape: {trends_synth.shape}")

wow_synth = compute_wow_kpis(clean_synth, complete_weeks=cw_synth)
print(f"WoW current_week: {wow_synth['current_week']}")

tp_synth = compute_team_performance(clean_synth)
print(f"Teams: {len(tp_synth)} rows")

bot_synth = compute_chatbot_escalation_analysis(clean_synth)
print(f"Chatbot esc rate: {bot_synth['overall_escalation_rate']:.3f}")

print("\n=== SYNTHETIC DATA OK ===")
