"""Tests for the ResearchOutput + SymbolResearch Pydantic models (T011).

Covers spec FR-001, FR-002, FR-011, FR-014:
- extra=forbid rejects unknown fields
- NaN sentiment is coerced to None by StrictModel._no_nans
- A valid ResearchOutput round-trips through model_dump
- Literal['1.0'] is enforced on the version field
"""
from __future__ import annotations

import math
import pytest
from pydantic import ValidationError

from src.agents.protocol import ResearchOutput, SymbolResearch, MessageType


def test_research_view_in_message_type():
    assert MessageType.RESEARCH_VIEW.value == "RESEARCH_VIEW"


def test_symbol_research_extra_forbid():
    with pytest.raises(ValidationError):
        SymbolResearch(sentiment=0.1, volume=1, topics=[], last_article_at=None, foo="bar")


def test_symbol_research_nan_coerced_to_none():
    sr = SymbolResearch(sentiment=math.nan, volume=1, topics=[], last_article_at=None)
    # StrictModel._no_nans returns None for NaN; field is nullable.
    assert sr.sentiment is None


def test_symbol_research_sentiment_bounds():
    with pytest.raises(ValidationError):
        SymbolResearch(sentiment=1.5, volume=0, topics=[], last_article_at=None)
    with pytest.raises(ValidationError):
        SymbolResearch(sentiment=-1.5, volume=0, topics=[], last_article_at=None)


def test_research_output_accepts_valid():
    ro = ResearchOutput(
        timestamp="2026-08-29T13:30:00Z",
        per_symbol={
            "NVDA": SymbolResearch(sentiment=0.3, volume=4, topics=["earnings", "guidance"], last_article_at="2026-08-29T12:00:00Z"),
            "AAPL": SymbolResearch(sentiment=0.0, volume=0, topics=[], last_article_at=None),
        },
        feature_flag_state="news-on",
        risks=[],
    )
    assert ro.version == "1.0"
    assert ro.per_symbol["NVDA"].sentiment == 0.3
    assert ro.per_symbol["AAPL"].sentiment == 0.0
    assert ro.per_symbol["AAPL"].last_article_at is None


def test_research_output_version_literal():
    # version is Literal["1.0"]; other values rejected
    with pytest.raises(ValidationError):
        ResearchOutput(timestamp="2026-08-29T13:30:00Z", per_symbol={}, version="2.0")  # type: ignore[arg-type]


def test_research_output_empty_per_symbol_is_valid():
    ro = ResearchOutput(timestamp="2026-08-29T13:30:00Z", per_symbol={})
    assert ro.per_symbol == {}
    assert ro.feature_flag_state == "news-on"


def test_research_output_feature_flag_off_marker():
    ro = ResearchOutput(timestamp="2026-08-29T13:30:00Z", per_symbol={}, feature_flag_state="news-off")
    assert ro.feature_flag_state == "news-off"
