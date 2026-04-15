"""Quick test script: run the full LangGraph pipeline end-to-end."""

import logging
import sys
import time
import traceback

sys.path.insert(0, "src")

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

from src.agent.graph import run_pipeline  # noqa: E402

try:
    start = time.time()
    result = run_pipeline()
    elapsed = time.time() - start

    print(f"\n=== Pipeline Complete in {elapsed:.1f}s ===")
    print("Execution log:")
    for entry in result.get("execution_log", []):
        print(f"  - {entry}")

    # Errors
    errors = result.get("errors", [])
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  ! {e}")

    opps = result.get("opportunities", [])
    print(f"\nOpportunities: {len(opps)}")
    for opp in opps[:5]:
        title = opp.get("title", "N/A")
        impact = opp.get("estimated_annual_impact_usd", opp.get("impact_estimate", "N/A"))
        print(f"  [{opp.get('rank', '?')}] {title} -- ${impact}")

    report = result.get("report_markdown", "")
    print(f"\nReport length: {len(report)} chars")
    print("Report preview:")
    print(report[:1000])

    # Quick state check
    for key in ["kpi_summary", "weekly_trends", "team_performance",
                 "channel_performance", "category_performance",
                 "anomalies", "chatbot_escalation", "nlp_summary",
                 "data_quality"]:
        val = result.get(key)
        status = "OK" if val else "MISSING"
        print(f"  state[{key}]: {status}")

except Exception:
    traceback.print_exc()
    sys.exit(1)
