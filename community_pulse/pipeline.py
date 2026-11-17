"""End-to-end run: records + events -> sentiment, topics, series, event shifts."""
from __future__ import annotations

from . import analysis as A
from . import models


def run(records: list[dict], events: list[dict], game: str, k: int = 8, window_days: int = 7,
        chat=None, embed=None, progress=print) -> dict:
    chat = chat or models.chat_json
    embed = embed or models.embed
    records = [dict(r) for r in records if r["text"].strip()]

    progress(f"sentiment: {len(records)} posts")
    for r, s in zip(records, models.llm_sentiment([r["text"] for r in records], game, chat=chat)):
        r["sentiment"], r["confidence"] = s["label"], s["confidence"]
        r["lexicon"] = models.lexicon_sentiment(r["text"])["label"]

    long = [r for r in records if len(r["text"].split()) >= A.MIN_WORDS]
    progress(f"topics: embedding {len(long)} posts with >= {A.MIN_WORDS} words")
    vecs = embed([r["text"] for r in long])
    assign = A.kmeans(vecs, k)
    topics = A.label_topics(long, assign, vecs, game, chat)
    for r in records:
        r["topic"] = A.NO_TOPIC
    for r, a in zip(long, assign):
        r["topic"] = a

    lo, hi = min(r["ts"] for r in records), max(r["ts"] for r in records)
    evs = [e for e in A.merge_events(events) if lo <= e["ts"] <= hi]
    for e in evs:
        e["shift"] = A.event_shift(records, e["ts"], topics, window_days)

    return {
        "game": game,
        "n_posts": len(records),
        "n_topic_posts": len(long),
        "range": [lo, hi],
        "topics": {str(t): v for t, v in topics.items()},
        "series_day": A.time_series(records, "day"),
        "series_week": A.time_series(records, "week"),
        "events": evs,
        "agreement_with_steam_vote": vote_agreement(records),
        "posts": [{k_: r[k_] for k_ in ("id", "ts", "text", "score", "voted_up", "sentiment", "confidence",
                                         "lexicon", "topic")} for r in records],
    }


def vote_agreement(records: list[dict]) -> dict | None:
    """How often a polar sentiment label matches the reviewer's own thumbs up/down.

    Not ground truth (people recommend games they complain about), but a large-sample check.
    """
    out = {}
    for key in ("sentiment", "lexicon"):
        polar = [r for r in records if r.get("voted_up") is not None and r[key] != "neutral"]
        if not polar:
            return None
        hit = sum((r[key] == "positive") == r["voted_up"] for r in polar)
        out[key] = {"n_polar": len(polar), "agree": round(hit / len(polar), 3),
                    "neutral_share": round(1 - len(polar) / len(records), 3)}
    return out
