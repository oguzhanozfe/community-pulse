"""Sentiment accuracy on the hand-labelled set: LLM vs lexicon baseline."""
from __future__ import annotations

import json
import math
import pathlib

from . import models

LABELS = models.LABELS


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p on discordant pairs (b: only A right, c: only B right)."""
    n = b + c
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def score(gold: list[str], pred: list[str]) -> dict:
    n = len(gold)
    hit = sum(g == p for g, p in zip(gold, pred))
    f1s = []
    for lab in LABELS:
        tp = sum(g == p == lab for g, p in zip(gold, pred))
        fp = sum(p == lab != g for g, p in zip(gold, pred))
        fn = sum(g == lab != p for g, p in zip(gold, pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    confusion = {g: {p: sum(gg == g and pp == p for gg, pp in zip(gold, pred)) for p in LABELS} for g in LABELS}
    return {"n": n, "accuracy": round(hit / n, 3), "accuracy_ci95": wilson(hit, n),
            "macro_f1": round(sum(f1s) / len(f1s), 3), "confusion_gold_x_pred": confusion}


def run(labels_path: str, chat=None) -> dict:
    items = [json.loads(line) for line in pathlib.Path(labels_path).read_text().splitlines() if line.strip()]
    gold = [it["label"] for it in items]
    llm = models.llm_sentiment([it["text"] for it in items], game="a video game", chat=chat)
    lex = [models.lexicon_sentiment(it["text"]) for it in items]
    llm_pred, lex_pred = [x["label"] for x in llm], [x["label"] for x in lex]
    b = sum(l == g != x for g, l, x in zip(gold, llm_pred, lex_pred))
    c = sum(x == g != l for g, l, x in zip(gold, llm_pred, lex_pred))
    calib = {}
    for lo, hi in ((0, 0.7), (0.7, 0.9), (0.9, 1.01)):
        sel = [g == p["label"] for g, p in zip(gold, llm) if lo <= p["confidence"] < hi]
        calib[f"{lo:.1f}-{min(hi, 1):.1f}"] = {"n": len(sel), "accuracy": round(sum(sel) / len(sel), 3) if sel else None}
    return {
        "chat_model": models.CHAT_MODEL,
        "models": models.model_info(),
        "gold_distribution": {lab: gold.count(lab) for lab in LABELS},
        "llm": score(gold, llm_pred),
        "lexicon": score(gold, lex_pred),
        "majority_class": score(gold, [max(LABELS, key=gold.count)] * len(gold)),
        "llm_vs_lexicon_mcnemar_p": round(mcnemar_exact(b, c), 4),
        "discordant": {"llm_only_right": b, "lexicon_only_right": c},
        "llm_calibration": calib,
        "items": [{"id": it["id"], "gold": g, "llm": l["label"], "llm_conf": l["confidence"], "lexicon": x}
                  for it, g, l, x in zip(items, gold, llm, lex_pred)],
    }


def summary_line(res: dict) -> str:
    l, x = res["llm"], res["lexicon"]
    return (f"On {l['n']} hand-labelled posts, {res['chat_model']} scored {l['accuracy'] * 100:.0f}% accuracy "
            f"(95% CI {l['accuracy_ci95'][0] * 100:.0f}-{l['accuracy_ci95'][1] * 100:.0f}%, macro-F1 {l['macro_f1']:.2f}) "
            f"vs {x['accuracy'] * 100:.0f}% for the lexicon baseline (macro-F1 {x['macro_f1']:.2f}).")
