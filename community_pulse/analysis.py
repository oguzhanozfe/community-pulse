"""Topic clustering, time series, and before/after event windows. Pure stdlib."""
from __future__ import annotations

import datetime as dt
import math
import random
import re
from collections import Counter, defaultdict

DAY = 86400
MIN_WORDS = 8          # shorter posts ("good game") carry sentiment but no topic
MIN_N = 30             # per-window sample below this -> "too few posts"
NO_TOPIC = -1
SENT_VALUE = {"positive": 1, "negative": -1, "neutral": 0}


# ------------------------------------------------------------------ k-means

def _norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def kmeans(vectors: list[list[float]], k: int, iters: int = 30, seed: int = 0) -> list[int]:
    """Spherical k-means (cosine) with k-means++ init. Deterministic for a given seed.

    ponytail: O(n*k*d) pure Python, fine to a few thousand posts; use numpy/sklearn beyond that.
    """
    rng = random.Random(seed)
    X = [_norm(v) for v in vectors]
    k = min(k, len(X))
    cents = [X[rng.randrange(len(X))]]
    while len(cents) < k:
        d = [min(1 - _dot(x, c) for c in cents) for x in X]
        total = sum(d)
        if total <= 0:
            break
        r, acc = rng.random() * total, 0.0
        for i, di in enumerate(d):
            acc += di
            if acc >= r:
                cents.append(X[i])
                break
    assign = [0] * len(X)
    for _ in range(iters):
        new = [max(range(len(cents)), key=lambda j: _dot(x, cents[j])) for x in X]
        if new == assign and _ > 0:
            break
        assign = new
        for j in range(len(cents)):
            members = [X[i] for i, a in enumerate(assign) if a == j]
            if members:
                cents[j] = _norm([sum(col) for col in zip(*members)])
    return assign


WORD = re.compile(r"[a-z][a-z'\-]{2,}")
STOP = set("""the and for that this with you but are was have not they just its it's all can get has had out
one what when your from more like about than them some there been would will their only even also into
very really game games play played playing don't dont i'm ive i've because much any after which who now
how time still were then other could make made being over these those most here want way see does did
lot thing things well back too off got get many every people should while where why our ever""".split())


def top_terms(texts_by_cluster: dict[int, list[str]], n: int = 8) -> dict[int, list[str]]:
    """Distinctive words per cluster (class-based tf-idf), for transparency next to LLM labels."""
    tf = {c: Counter(w for t in ts for w in set(WORD.findall(t.lower())) if w not in STOP)
          for c, ts in texts_by_cluster.items()}
    df = Counter(w for c in tf for w in tf[c])
    K = len(tf)
    return {c: [w for w, _ in sorted(((w, f * math.log(1 + K / df[w])) for w, f in cnt.items() if f >= 3),
                                     key=lambda x: -x[1])[:n]] for c, cnt in tf.items()}


TOPIC_PROMPT = """These are player reviews of "{game}" that an embedding model grouped together.
Frequent distinctive words: {terms}
Reviews:
{samples}
Name the shared topic in 2-4 words (e.g. "Anti-cheat and kicks", "Story and characters") and describe it in one short sentence.
Return JSON only: {{"label": "...", "description": "..."}}"""


def label_topics(records, assign, vectors, game, chat, samples=12) -> dict[int, dict]:
    """LLM label per cluster from the posts closest to its centroid."""
    by = defaultdict(list)
    for r, a, v in zip(records, assign, vectors):
        by[a].append((r, _norm(v)))
    terms = top_terms({c: [r["text"] for r, _ in rs] for c, rs in by.items()})
    out = {}
    for c, rs in sorted(by.items()):
        cent = _norm([sum(col) for col in zip(*[v for _, v in rs])])
        near = sorted(rs, key=lambda rv: -_dot(rv[1], cent))[:samples]
        text = "\n".join(f"- {' '.join(r['text'].split())[:300]}" for r, _ in near)
        try:
            got = chat(TOPIC_PROMPT.format(game=game, terms=", ".join(terms[c]), samples=text))
            label, desc = str(got["label"]).strip(), str(got.get("description", "")).strip()
        except (KeyError, TypeError, ValueError):
            label, desc = f"Topic {c}", ""
        out[c] = {"label": label or f"Topic {c}", "description": desc, "terms": terms[c], "size": len(rs)}
    return out


# ------------------------------------------------------------- time series

def bucket(ts: int, freq: str) -> str:
    d = dt.datetime.fromtimestamp(ts, dt.timezone.utc).date()
    if freq == "week":
        d -= dt.timedelta(days=d.weekday())
    return d.isoformat()


def time_series(records: list[dict], freq: str = "day") -> dict:
    """{bucket: {"n", "net", "topics": {topic: {"n", "net"}}}}; net = mean of +1/0/-1 sentiment."""
    acc = defaultdict(lambda: {"n": 0, "s": 0, "topics": defaultdict(lambda: [0, 0])})
    for r in records:
        b = acc[bucket(r["ts"], freq)]
        v = SENT_VALUE[r["sentiment"]]
        b["n"] += 1
        b["s"] += v
        t = b["topics"][r["topic"]]
        t[0] += 1
        t[1] += v
    return {k: {"n": b["n"], "net": round(b["s"] / b["n"], 3),
                "topics": {t: {"n": n, "net": round(s / n, 3)} for t, (n, s) in sorted(b["topics"].items())}}
            for k, b in sorted(acc.items())}


# --------------------------------------------------------- event windows

def two_prop_p(x1: int, n1: int, x2: int, n2: int) -> float | None:
    """Two-sided p-value of a two-proportion z-test (normal approximation)."""
    if not n1 or not n2:
        return None
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (x2 / n2 - x1 / n1) / se
    return math.erfc(abs(z) / math.sqrt(2))


def fisher_p(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-sided Fisher exact p for a 2x2 table; used when a cell is under 5."""
    k, n = x1 + x2, n1 + n2
    total = math.comb(n, k)
    pmf = lambda a: math.comb(n1, a) * math.comb(n2, k - a) / total
    p0 = pmf(x1)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(pmf(a) for a in range(lo, hi + 1) if pmf(a) <= p0 * (1 + 1e-9)))


def prop_test(x1: int, n1: int, x2: int, n2: int) -> float | None:
    if not n1 or not n2:
        return None
    if min(x1, x2, n1 - x1, n2 - x2) < 5:
        return fisher_p(x1, n1, x2, n2)
    return two_prop_p(x1, n1, x2, n2)


def verdict(p_adj, n1, n2) -> str:
    if min(n1, n2) < MIN_N:
        return "too few posts"
    return "likely shift" if p_adj is not None and p_adj < 0.05 else "no clear shift"


def merge_events(events: list[dict], min_gap_days: float = 3) -> list[dict]:
    """Collapse announcements within `min_gap_days` (e.g. patch + hotfix) into one event."""
    out = []
    for e in sorted(events, key=lambda e: e["ts"]):
        if out and e["ts"] - out[-1]["ts"] < min_gap_days * DAY:
            out[-1]["title"] += " + " + e["title"]
        else:
            out.append(dict(e))
    return out


def event_shift(records: list[dict], event_ts: int, topics: dict, window_days: int = 7) -> dict:
    """Compare the `window_days` before vs after an event: negative share and each topic's share.

    Two-proportion z-test, or Fisher's exact test when a cell is under 5;
    p-values are Bonferroni-adjusted across the tests run for this event. Proportions are
    of *sampled* posts, so they show mix and mood, not volume.
    """
    w = window_days * DAY
    before = [r for r in records if event_ts - w <= r["ts"] < event_ts]
    after = [r for r in records if event_ts <= r["ts"] < event_ts + w]
    n1, n2 = len(before), len(after)
    tests = []

    def add(name, f, kind):
        x1, x2 = sum(map(f, before)), sum(map(f, after))
        tests.append({"name": name, "kind": kind, "before": x1, "after": x2,
                      "share_before": round(x1 / n1, 3) if n1 else None,
                      "share_after": round(x2 / n2, 3) if n2 else None,
                      "p": prop_test(x1, n1, x2, n2)})

    add("negative sentiment", lambda r: r["sentiment"] == "negative", "sentiment")
    if all(r.get("voted_up") is not None for r in before + after):
        add("not recommended (Steam vote)", lambda r: not r["voted_up"], "sentiment")
    for t, info in sorted(topics.items()):
        add(info["label"], lambda r, t=t: r["topic"] == t, "topic")
    m = len(tests)
    for t in tests:
        t["p_adj"] = None if t["p"] is None else min(1.0, t["p"] * m)
        t["verdict"] = verdict(t["p_adj"], n1, n2)
        t["p"] = None if t["p"] is None else round(t["p"], 4)
        t["p_adj"] = None if t["p_adj"] is None else round(t["p_adj"], 4)
    return {"n_before": n1, "n_after": n2, "window_days": window_days, "tests": tests}
