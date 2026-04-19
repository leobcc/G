"""NLP analysis module for ticket text mining.

Provides five-level sentiment analysis using VADER (Valence Aware Dictionary
and sEntiment Reasoner), regex-based frustration detection, and TF-IDF
topic extraction.

VADER is the gold-standard for social-media / short-text sentiment:
- Pure Python (~140KB lexicon), no heavy models, no API calls
- Processes 10K messages in < 1 second
- Returns a compound score on [-1, 1] mapped to 5 levels:
  very_negative / negative / neutral / positive / very_positive
- Human-validated lexicon handles slang, emoji, punctuation emphasis

No Groq or external API calls are made for sentiment — Groq is reserved
exclusively for reports and executive insights.
"""

import logging
import re

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import FRUSTRATION_PATTERNS

logger = logging.getLogger(__name__)

# Compiled frustration regex for performance
_frustration_regex = re.compile(
    "|".join(FRUSTRATION_PATTERNS), re.IGNORECASE
)

# ---------------------------------------------------------------------------
# VADER sentiment — singleton analyser
# ---------------------------------------------------------------------------
_vader = SentimentIntensityAnalyzer()

# Threshold mapping:  compound score → 5-level label
#   [-1.0, -0.5)  → very_negative
#   [-0.5, -0.05) → negative
#   [-0.05, 0.05] → neutral
#   ( 0.05, 0.5]  → positive
#   ( 0.5,  1.0]  → very_positive
_SENTIMENT_THRESHOLDS: list[tuple[float, str]] = [
    (-0.5, "very_negative"),
    (-0.05, "negative"),
    (0.05, "neutral"),
    (0.5, "positive"),
]


def _compound_to_label(compound: float) -> str:
    """Map VADER compound score to a 5-level sentiment label."""
    for threshold, label in _SENTIMENT_THRESHOLDS:
        if compound < threshold:
            return label
    return "very_positive"


def compute_sentiment(text: str) -> dict:
    """Compute sentiment for a single text using VADER.

    Returns a normalised compound score on [-1, 1] and a 5-level label.

    Args:
        text: Input text string.

    Returns:
        Dict with polarity (-1 to 1), confidence (0 to 1), and label.
    """
    if not isinstance(text, str) or not text.strip():
        return {"polarity": 0.0, "confidence": 0.0, "label": "neutral"}

    scores = _vader.polarity_scores(text)
    compound = scores["compound"]
    label = _compound_to_label(compound)
    # Confidence = absolute compound (higher magnitude → more confident)
    confidence = min(abs(compound) * 1.2, 1.0)
    return {
        "polarity": round(compound, 4),
        "confidence": round(confidence, 4),
        "label": label,
    }


def add_sentiment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add 5-level sentiment columns to every row using VADER.

    VADER processes 10K messages in < 1 second — no sampling needed.

    Columns added:
        - sentiment_polarity: compound score [-1, 1]
        - sentiment_confidence: abs(compound) scaled [0, 1]
        - sentiment_label: one of very_negative / negative / neutral /
          positive / very_positive

    Args:
        df: DataFrame with 'customer_message' column.

    Returns:
        DataFrame with sentiment columns added.
    """
    df = df.copy()
    logger.info("Running VADER sentiment on %d messages ...", len(df))

    sentiments = df["customer_message"].apply(compute_sentiment)
    df["sentiment_polarity"] = sentiments.apply(lambda x: x["polarity"])
    df["sentiment_confidence"] = sentiments.apply(lambda x: x["confidence"])
    df["sentiment_label"] = sentiments.apply(lambda x: x["label"])

    logger.info(
        "Sentiment complete: %s",
        df["sentiment_label"].value_counts().to_dict(),
    )
    return df


def detect_frustration(text: str) -> dict:
    """Detect frustration signals in customer text.

    Args:
        text: Input text string.

    Returns:
        Dict with is_frustrated bool, matched patterns, and frustration score.
    """
    if not isinstance(text, str) or not text.strip():
        return {"is_frustrated": False, "matched_patterns": [], "score": 0.0}

    matches = [m.group() for m in _frustration_regex.finditer(text.lower())]
    # Count exclamation mark clusters as frustration signal
    exclamation_count = len(re.findall(r"!{2,}", text))
    caps_ratio = (
        sum(1 for c in text if c.isupper()) / max(len(text), 1)
    )

    base_score = len(set(matches)) * 0.3
    exclamation_score = min(exclamation_count * 0.15, 0.3)
    caps_score = max(0, (caps_ratio - 0.3)) * 0.5 if caps_ratio > 0.3 else 0

    score = min(base_score + exclamation_score + caps_score, 1.0)

    return {
        "is_frustrated": score >= 0.3 or len(matches) >= 1,
        "matched_patterns": list(set(matches)),
        "score": round(score, 2),
    }


def add_frustration_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add frustration detection columns to DataFrame.

    Args:
        df: DataFrame with 'customer_message' column.

    Returns:
        DataFrame with 'is_frustrated' and 'frustration_score' columns.
    """
    df = df.copy()
    frustration = df["customer_message"].apply(detect_frustration)
    df["is_frustrated"] = frustration.apply(lambda x: x["is_frustrated"])
    df["frustration_score"] = frustration.apply(lambda x: x["score"])
    return df


def extract_topics(
    df: pd.DataFrame,
    text_column: str = "customer_message",
    n_topics: int = 8,
    n_top_words: int = 10,
    max_features: int = 1000,
) -> dict:
    """Extract dominant topics from ticket text using TF-IDF + KMeans.

    Args:
        df: DataFrame with text column.
        text_column: Name of the text column.
        n_topics: Number of topic clusters.
        n_top_words: Number of top words per topic.
        max_features: Maximum TF-IDF features.

    Returns:
        Dict with topics list and cluster assignments.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = df[text_column].fillna("").astype(str)
    texts = texts[texts.str.len() > 10]  # Skip very short texts

    if len(texts) < n_topics * 2:
        logger.warning("Not enough texts for %d topics", n_topics)
        return {"topics": [], "assignments": []}

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=3,
        max_df=0.8,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    kmeans = MiniBatchKMeans(n_clusters=n_topics, random_state=42, n_init=3)
    kmeans.fit(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()

    # Assign clusters to all rows (including short texts)
    full_tfidf = vectorizer.transform(df[text_column].fillna("").astype(str))
    assignments = kmeans.predict(full_tfidf).tolist()

    # Count tickets per cluster
    cluster_counts = pd.Series(assignments).value_counts().to_dict()

    topics = []
    for i, centroid in enumerate(kmeans.cluster_centers_):
        top_indices = centroid.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[j] for j in top_indices]
        topics.append(
            {
                "topic_id": i,
                "top_words": top_words,
                "count": cluster_counts.get(i, 0),
                "label": f"Topic {i}",
            }
        )

    return {
        "topics": topics,
        "assignments": assignments,
    }


def compute_nlp_summary(df: pd.DataFrame) -> dict:
    """Run full NLP pipeline and return summary statistics.

    Includes VADER sentiment (5-level), frustration detection,
    topic clustering, and cross-dimensional breakdowns for executive insights.

    For datasets > 50K rows, NLP is run on a representative sample of 10K rows
    to keep processing time under control (VADER ≈ 1s/10K, frustration ≈ instant).

    Args:
        df: DataFrame with 'customer_message' column.

    Returns:
        Dict with sentiment distribution, frustration stats, topic summary,
        and per-dimension breakdowns.
    """
    # For large datasets, sample to keep NLP fast
    MAX_NLP_ROWS = 50_000
    sampled = False
    if len(df) > MAX_NLP_ROWS:
        logger.info(
            "Dataset has %d rows — sampling %d for NLP analysis",
            len(df), MAX_NLP_ROWS,
        )
        df = df.sample(n=MAX_NLP_ROWS, random_state=42).reset_index(drop=True)
        sampled = True
    # Sentiment (VADER — 5-level, pure Python, no API calls)
    if "customer_message" not in df.columns:
        logger.warning("No 'customer_message' column — skipping NLP analysis")
        return {
            "avg_sentiment_polarity": 0.0,
            "avg_sentiment_confidence": 0.0,
            "sentiment_distribution": {},
            "total_frustrated_tickets": 0,
            "frustration_rate": 0.0,
            "frustration_by_category": {},
            "sentiment_by_team": {},
            "sentiment_by_channel": {},
            "sentiment_by_priority": {},
            "csat_by_frustration": {},
            "topics": [],
            "sample_frustrated_messages": [],
        }

    df = add_sentiment_columns(df)
    df = add_frustration_columns(df)

    # Topic extraction
    topic_results = extract_topics(df)

    # Sentiment distribution (5 levels: very_negative … very_positive)
    sentiment_dist = df["sentiment_label"].value_counts().to_dict()

    # Frustration stats by category
    frustration_by_category = (
        df.groupby("category")["is_frustrated"]
        .mean()
        .round(3)
        .sort_values(ascending=False)
        .to_dict()
    )

    # ── Rich cross-dimensional breakdowns for executive insights ──

    # Sentiment by team
    sentiment_by_team = {}
    if "assigned_team" in df.columns:
        sentiment_by_team = (
            df.groupby("assigned_team")
            .agg(
                avg_polarity=("sentiment_polarity", "mean"),
                negative_pct=("sentiment_label", lambda s: s.isin(["negative", "very_negative"]).mean()),
                positive_pct=("sentiment_label", lambda s: s.isin(["positive", "very_positive"]).mean()),
                frustration_rate=("is_frustrated", "mean"),
            )
            .round(3)
            .to_dict("index")
        )

    # Sentiment by channel
    sentiment_by_channel = {}
    if "channel" in df.columns:
        sentiment_by_channel = (
            df.groupby("channel")
            .agg(
                avg_polarity=("sentiment_polarity", "mean"),
                negative_pct=("sentiment_label", lambda s: s.isin(["negative", "very_negative"]).mean()),
                positive_pct=("sentiment_label", lambda s: s.isin(["positive", "very_positive"]).mean()),
                frustration_rate=("is_frustrated", "mean"),
            )
            .round(3)
            .to_dict("index")
        )

    # Sentiment by priority
    sentiment_by_priority = {}
    if "priority" in df.columns:
        sentiment_by_priority = (
            df.groupby("priority")
            .agg(
                avg_polarity=("sentiment_polarity", "mean"),
                negative_pct=("sentiment_label", lambda s: s.isin(["negative", "very_negative"]).mean()),
                frustration_rate=("is_frustrated", "mean"),
            )
            .round(3)
            .to_dict("index")
        )

    # Frustration-CSAT correlation (frustrated tickets → lower CSAT?)
    csat_by_frustration = {}
    if "csat_score" in df.columns:
        valid_csat = df.dropna(subset=["csat_score"])
        if not valid_csat.empty:
            csat_by_frustration = {
                "frustrated_avg_csat": round(
                    valid_csat.loc[valid_csat["is_frustrated"], "csat_score"].mean(), 2
                ),
                "non_frustrated_avg_csat": round(
                    valid_csat.loc[~valid_csat["is_frustrated"], "csat_score"].mean(), 2
                ),
            }

    # Sample frustrated messages
    frustrated_msgs = df.loc[
        df["is_frustrated"], "customer_message"
    ].dropna().head(10).tolist()

    return {
        "avg_sentiment_polarity": round(float(df["sentiment_polarity"].mean()), 4),
        "avg_sentiment_confidence": round(float(df["sentiment_confidence"].mean()), 4),
        "sentiment_distribution": sentiment_dist,
        "total_frustrated_tickets": int(df["is_frustrated"].sum()),
        "frustration_rate": round(float(df["is_frustrated"].mean()), 4),
        "frustration_by_category": frustration_by_category,
        "sentiment_by_team": sentiment_by_team,
        "sentiment_by_channel": sentiment_by_channel,
        "sentiment_by_priority": sentiment_by_priority,
        "csat_by_frustration": csat_by_frustration,
        "topics": topic_results["topics"],
        "sample_frustrated_messages": frustrated_msgs,
    }
