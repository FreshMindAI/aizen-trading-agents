"""Built-in finance-tuned sentiment lexicon (spec 003 / T007, research Decision 1).

Loughran-McDonald (~4,000 words) is the canonical finance-NLP lexicon, but it
ships as a 1.2 MB CSV that is not in the repo. This module is a smaller,
built-in fallback (~100 positive / ~100 negative words) that is always
available and runs in <100ms for 50 articles.

The contract is deterministic and bounded in [-1, +1]:

    sentiment = (positive_hits - negative_hits) / max(1, total_words)

The caller is responsible for tokenization (we use a simple lowercase +
word-boundary regex here so the lexicon is self-contained).
"""
from __future__ import annotations

import re
from functools import lru_cache

# ---------------------------------------------------------------------------
# Word lists (curated; extend via LM CSV when available)
# ---------------------------------------------------------------------------
POSITIVE: frozenset[str] = frozenset({
    "beat", "beats", "beating", "raised", "raise", "raises", "raising",
    "surge", "surges", "surged", "surging", "rally", "rallies", "rallied",
    "rallying", "gain", "gains", "gained", "gaining", "upbeat", "optimistic",
    "strong", "stronger", "strongest", "robust", "solid", "exceed",
    "exceeded", "exceeds", "exceeding", "outperform", "outperformed",
    "outperforms", "outperforming", "growth", "growing", "grew", "grows",
    "expand", "expands", "expanded", "expanding", "expansion", "upgrade",
    "upgraded", "upgrades", "upgrading", "win", "wins", "won", "winning",
    "breakthrough", "breakthroughs", "record", "records", "record-high",
    "high", "highs", "soar", "soars", "soared", "soaring", "boom", "booms",
    "booming", "boomed", "bullish", "bull", "bulls", "approve", "approved",
    "approves", "approving", "approval", "launch", "launches", "launched",
    "launching", "profit", "profits", "profitable", "positive", "benefit",
    "benefits", "benefited", "benefiting", "success", "successful",
    "succeed", "succeeded", "succeeds", "succeeding", "achieve", "achieved",
    "achieves", "achieving", "milestone", "milestones", "innovation",
    "innovate", "innovated", "innovating", "innovative", "innovator",
    "leadership", "leading", "leader", "leaders", "advance", "advances",
    "advanced", "advancing", "progress", "progresses", "progressed",
    "progressing", "opportunity", "opportunities", "demand", "demands",
    "demanded", "demanding", "momentum", "accelerate", "accelerates",
    "accelerated", "accelerating", "win-win", "tailwind", "tailwinds",
    "outpace", "outpaced", "outpaces", "outpacing",
})

NEGATIVE: frozenset[str] = frozenset({
    "miss", "missed", "misses", "missing", "cut", "cuts", "cutting",
    "plunge", "plunges", "plunged", "plunging", "plummet", "plummets",
    "plummeted", "plummeting", "tumble", "tumbles", "tumbled", "tumbling",
    "drop", "drops", "dropped", "dropping", "fall", "falls", "fell",
    "falling", "decline", "declines", "declined", "declining", "downgrade",
    "downgraded", "downgrades", "downgrading", "warn", "warns", "warned",
    "warning", "weak", "weaker", "weakest", "weakness", "concern", "concerns",
    "concerned", "concerning", "risk", "risks", "risky", "loss", "losses",
    "lose", "loses", "lost", "losing", "negative", "underperform",
    "underperformed", "underperforms", "underperforming", "shrink", "shrinks",
    "shrunk", "shrinking", "shrinkage", "layoff", "layoffs", "fire", "fires",
    "fired", "firing", "halt", "halts", "halted", "halting", "suspend",
    "suspends", "suspended", "suspending", "fraud", "frauds", "fraudulent",
    "scandal", "scandals", "lawsuit", "lawsuits", "probe", "probes",
    "probed", "probing", "investigation", "investigations", "investigate",
    "investigated", "investigating", "subpoena", "subpoenas", "penalty",
    "penalties", "fine", "fines", "fined", "violation", "violations",
    "breach", "breaches", "breached", "breaching", "default", "defaults",
    "defaulted", "defaulting", "bankruptcy", "bankrupt", "insolvent",
    "insolvency", "recall", "recalls", "recalled", "recalling", "crash",
    "crashes", "crashed", "crashing", "slide", "slides", "slid", "sliding",
    "slump", "slumps", "slumped", "slumping", "bearish", "bear", "bears",
    "selloff", "sell-offs", "headwind", "headwinds", "obstacle", "obstacles",
    "setback", "setbacks", "disappointing", "disappointed", "disappoints",
    "disappoint", "recession", "recessions", "inflation", "stagflation",
    "lay-offs", "downturn", "downturns",
})


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']{1,}")


@lru_cache(maxsize=2048)
def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(t.lower() for t in _TOKEN_RE.findall(text or ""))


def lexicon_sentiment(text: str) -> float:
    """Return sentiment in [-1, +1] for ``text``.

    Formula (spec FR-011):
        (positive_hits - negative_hits) / max(1, total_words)
    """
    if not text:
        return 0.0
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in POSITIVE)
    neg = sum(1 for t in tokens if t in NEGATIVE)
    return (pos - neg) / max(1, len(tokens))


def per_article_sentiment(article: dict) -> float:
    """Compute sentiment for a single Alpaca news article dict.

    Uses ``headline`` + ``summary`` (when present) and ``content`` (when
    ``include_content`` was set on the fetch). Returns 0.0 if the article
    has no text fields.
    """
    parts: list[str] = []
    for k in ("headline", "summary"):
        v = article.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    content = article.get("content")
    if isinstance(content, str) and content.strip():
        parts.append(content)
    return lexicon_sentiment("\n".join(parts))


__all__ = [
    "POSITIVE",
    "NEGATIVE",
    "lexicon_sentiment",
    "per_article_sentiment",
]
