"""Tests for the NLP analysis module."""

import pytest

from src.nlp_analysis import compute_sentiment, detect_frustration


class TestSentiment:
    def test_positive_text(self):
        result = compute_sentiment("I love this service, it was wonderful!")
        assert result["polarity"] > 0

    def test_negative_text(self):
        result = compute_sentiment("This is terrible, worst experience ever")
        assert result["polarity"] < 0

    def test_empty_text(self):
        result = compute_sentiment("")
        assert result["polarity"] == 0.0


class TestFrustration:
    def test_frustrated_text(self):
        result = detect_frustration("This is ridiculous!! I've been waiting forever")
        assert result["is_frustrated"] is True
        assert len(result["matched_patterns"]) > 0

    def test_calm_text(self):
        result = detect_frustration("Could you please help me with my order?")
        assert result["is_frustrated"] is False

    def test_empty_text(self):
        result = detect_frustration("")
        assert result["is_frustrated"] is False
        assert result["score"] == 0.0
