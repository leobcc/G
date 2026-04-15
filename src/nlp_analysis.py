"""NLP analysis module for ticket text mining.

Provides sentiment analysis, topic extraction, and frustration detection
using lightweight libraries (TextBlob, scikit-learn TF-IDF).
"""

import logging
import re

import pandas as pd
from textblob import TextBlob

from src.config import FRUSTRATION_PATTERNS

logger = logging.getLogger(__name__)

# Compiled frustration regex for performance
_frustration_regex = re.compile(
    "|".join(FRUSTRATION_PATTERNS), re.IGNORECASE
)


def compute_sentiment(text: str) -> dict:
    """Compute sentiment polarity and subjectivity for a single text.

    Args:
        text: Input text string.

    Returns:
        Dict with polarity (-1 to 1) and subjectivity (0 to 1).
    """
    if not isinstance(text, str) or not text.strip():
        return {"polarity": 0.0, "subjectivity": 0.0}

    blob = TextBlob(text)
    return {
        "polarity": round(blob.sentiment.polarity, 3),
        "subjectivity": round(blob.sentiment.subjectivity, 3),
    }


def add_sentiment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment polarity and subjectivity columns to DataFrame.

    Uses the 'customer_message' column.

    Args:
        df: DataFrame with 'customer_message' column.

    Returns:
        DataFrame with 'sentiment_polarity' and 'sentiment_subjectivity' columns added.
    """
    df = df.copy()
    sentiments = df["customer_message"].apply(compute_sentiment)
    df["sentiment_polarity"] = sentiments.apply(lambda x: x["polarity"])
    df["sentiment_subjectivity"] = sentiments.apply(lambda x: x["subjectivity"])
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
    topics = []
    for i, centroid in enumerate(kmeans.cluster_centers_):
        top_indices = centroid.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[j] for j in top_indices]
        topics.append(
            {
                "topic_id": i,
                "top_words": top_words,
                "label": f"Topic {i}",  # Placeholder; LLM will generate labels
            }
        )

    # Assign clusters to all rows (including short texts)
    full_tfidf = vectorizer.transform(df[text_column].fillna("").astype(str))
    assignments = kmeans.predict(full_tfidf).tolist()

    return {
        "topics": topics,
        "assignments": assignments,
    }


def compute_nlp_summary(df: pd.DataFrame) -> dict:
    """Run full NLP pipeline and return summary statistics.

    Args:
        df: DataFrame with 'customer_message' column.

    Returns:
        Dict with sentiment distribution, frustration stats, topic summary.
    """
    # Sentiment
    df = add_sentiment_columns(df)
    df = add_frustration_columns(df)

    # Topic extraction
    topic_results = extract_topics(df)

    # Sentiment distribution buckets
    bins = [-1.01, -0.3, -0.05, 0.05, 0.3, 1.01]
    labels = ["very_negative", "negative", "neutral", "positive", "very_positive"]
    df["sentiment_bucket"] = pd.cut(
        df["sentiment_polarity"], bins=bins, labels=labels
    )
    sentiment_dist = df["sentiment_bucket"].value_counts().to_dict()

    # Frustration stats by category
    frustration_by_category = (
        df.groupby("category")["is_frustrated"]
        .mean()
        .round(3)
        .sort_values(ascending=False)
        .to_dict()
    )

    return {
        "avg_sentiment_polarity": round(df["sentiment_polarity"].mean(), 3),
        "avg_sentiment_subjectivity": round(df["sentiment_subjectivity"].mean(), 3),
        "sentiment_distribution": sentiment_dist,
        "total_frustrated_tickets": int(df["is_frustrated"].sum()),
        "frustration_rate": round(df["is_frustrated"].mean(), 3),
        "frustration_by_category": frustration_by_category,
        "topics": topic_results["topics"],
    }
