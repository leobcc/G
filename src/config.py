"""Central configuration for the Ops Intelligence system."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
RAW_DATA_PATH = DATA_DIR / "option_a_ticket_data.csv"

# ---------------------------------------------------------------------------
# Data constants
# ---------------------------------------------------------------------------
VALID_CHANNELS = ["email", "chat", "phone", "social"]
VALID_CATEGORIES = [
    "refund",
    "order_status",
    "merchant_issue",
    "voucher_problem",
    "billing",
    "account",
    "other",
]
VALID_PRIORITIES = ["low", "medium", "high", "urgent"]
VALID_TEAMS = ["in_house", "bpo_vendorA", "bpo_vendorB", "ai_chatbot"]
VALID_STATUSES = ["resolved", "escalated", "abandoned", "pending"]
VALID_MARKETS = ["US", "UK", "DE", "FR", "ES", "IT", "AU"]

MARKET_NORMALIZATION: dict[str, str] = {
    "United Kingdom": "UK",
    "GER": "DE",
    "USA": "US",
    "": "Unknown",
}

CSAT_MIN = 1
CSAT_MAX = 5

# Scale factor: sample ~10K tickets over 4 weeks; Groupon handles 120K/month
SCALE_FACTOR = 12

# Complete weeks in the dataset (Week 11 is partial — only 224 tickets)
COMPLETE_WEEKS = [7, 8, 9, 10]

# ---------------------------------------------------------------------------
# Branding
# ---------------------------------------------------------------------------
GROUPON_GREEN = "#53A318"
GROUPON_DARK = "#1A1A1A"
GROUPON_LIGHT_GREEN = "#7BC74D"

# Color palette for charts
CHART_COLORS = [
    "#53A318",  # Groupon green
    "#2196F3",  # Blue
    "#FF9800",  # Orange
    "#E91E63",  # Pink
    "#9C27B0",  # Purple
    "#00BCD4",  # Cyan
    "#FF5722",  # Deep orange
]

TEAM_COLORS = {
    "in_house": "#53A318",
    "bpo_vendorA": "#2196F3",
    "bpo_vendorB": "#FF9800",
    "ai_chatbot": "#9C27B0",
}

STATUS_COLORS = {
    "resolved": "#53A318",
    "escalated": "#FF9800",
    "abandoned": "#E91E63",
    "pending": "#9E9E9E",
}

# ---------------------------------------------------------------------------
# LLM settings (Google Gemini — free tier)
# ---------------------------------------------------------------------------
LLM_MODEL_ANALYSIS = "gemini-2.0-flash"
LLM_MODEL_SIMPLE = "gemini-2.0-flash-lite"
LLM_TEMPERATURE = 0.1  # Low temperature for analytical consistency

# ---------------------------------------------------------------------------
# Frustration detection patterns
# ---------------------------------------------------------------------------
FRUSTRATION_PATTERNS = [
    r"ridiculous",
    r"terrible",
    r"worst",
    r"horrible",
    r"unacceptable",
    r"still waiting",
    r"3rd time|third time",
    r"no one helps",
    r"never again",
    r"waste of",
    r"scam",
    r"fraud",
    r"stolen",
    r"rip.?off",
    r"\bugh+\b",
    r"\bargh+\b",
    r"!!+",
    r"\?\?+",
    r"this is (?:a )?joke",
    r"can't believe",
    r"extremely frustrated",
    r"disgusted",
    r"furious",
    r"livid",
    r"please\s+help\s+me\b.*!",  # "please help me!" with emphasis
]
