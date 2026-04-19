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
GROUPON_DARK = "#3d3a2a"          # warm dark olive (Anthropic-inspired)
GROUPON_LIGHT_GREEN = "#7BC74D"

# Warm theme palette (matches .streamlit/config.toml)
THEME_BG = "#fdfdf8"              # warm cream background
THEME_SECONDARY_BG = "#ecebe3"    # warm beige secondary
THEME_BORDER = "#d3d2ca"          # warm gray border
THEME_SIDEBAR_BG = "#f0f0ec"      # sidebar background

# ---------------------------------------------------------------------------
# Color palette — professional, cohesive palette anchored on Groupon green
# ---------------------------------------------------------------------------
CHART_COLORS = [
    "#53A318",  # Groupon green (primary)
    "#1B6B9A",  # Deep teal-blue
    "#E8793A",  # Warm tangerine
    "#6B4C9A",  # Muted purple
    "#2E9E8F",  # Sea green
    "#C44D56",  # Muted crimson
    "#4A7C3F",  # Forest green
]

TEAM_COLORS = {
    "in_house": "#53A318",    # Groupon green — best performer
    "bpo_vendorA": "#1B6B9A", # Deep teal-blue
    "bpo_vendorB": "#E8793A", # Warm tangerine
    "ai_chatbot": "#6B4C9A",  # Muted purple
}

STATUS_COLORS = {
    "resolved": "#53A318",   # Green — success
    "escalated": "#E8793A",  # Tangerine — warning
    "abandoned": "#C44D56",  # Crimson — danger
    "pending": "#8C8C8C",    # Neutral grey
}

# Sentiment-specific colors
SENTIMENT_COLORS = {
    "positive": "#53A318",
    "neutral": "#8C8C8C",
    "negative": "#C44D56",
}

# Sequential palette for heatmaps / gradients (low → high)
HEATMAP_SCALE = [
    [0, "#C44D56"],     # Bad (red tone)
    [0.5, "#F5D660"],   # Middle (warm yellow)
    [1, "#53A318"],     # Good (green)
]

# ---------------------------------------------------------------------------
# LLM settings (Groq — free tier, Llama 4 Scout: 500K TPD, 30K TPM, 30 RPM)
# ---------------------------------------------------------------------------
LLM_MODEL_ANALYSIS = "meta-llama/llama-4-scout-17b-16e-instruct"
LLM_MODEL_SIMPLE = "meta-llama/llama-4-scout-17b-16e-instruct"
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
