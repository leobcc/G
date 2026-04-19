"""Quick E2E pipeline test."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.agent.graph import run_pipeline

result = run_pipeline()
print("=== PIPELINE RESULT KEYS ===")
print(sorted(result.keys()))
print()

print("=== EXECUTIVE INSIGHTS ===")
ei = result.get("executive_insights", {})
print(f"Keys: {list(ei.keys())}")
for k, v in ei.items():
    preview = str(v)[:150] if v else "(empty)"
    print(f"  {k}: {preview}")
print()

opps = result.get("opportunities", [])
print(f"Opportunities: {len(opps)}")
brief = result.get("weekly_brief", "")
print(f"Report length: {len(brief)} chars")
log = result.get("pipeline_log", [])
print(f"Pipeline steps: {len(log)}")
for step in log:
    print(f"  - {step}")
print()
print("=== SUCCESS ===")
