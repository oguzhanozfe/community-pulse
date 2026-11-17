"""Data sources: Steam reviews, Steam news (patch dates), Reddit (official OAuth API).

Every source returns plain dicts with the same shape and never keeps author data:
    {"id", "source", "ts", "text", "score", "voted_up"}
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "community-pulse/0.1 (+https://github.com/oguzhanozfe/community-pulse)"
STEAM_REVIEWS = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"
STEAM_NEWS = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
MAX_TEXT = 2000
DAY = 86400


# Slurs are masked at ingest so committed examples don't republish them; they carry no topic signal.
SLURS = re.compile(r"\b(n[i1]gg+(?:a+h?|er|uh)s?|f[a@]gg?(?:ot)?s?|retard(?:ed|s)?|tr[a@]nn(?:y|ies))\b", re.I)


def clean_text(text: str) -> str:
    return SLURS.sub("[slur]", (text or "").strip())[:MAX_TEXT]


class RateLimiter:
    """Minimum spacing between calls; one instance per host."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self):
        delay = self._last + self.min_interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


def http_json(url, *, headers=None, data=None, limiter=None, retries=4):
    """GET/POST returning parsed JSON; backs off on 429/5xx, honours Retry-After."""
    for attempt in range(retries + 1):
        if limiter:
            limiter.wait()
        req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp), dict(resp.headers)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
            time.sleep(float(e.headers.get("Retry-After") or 2 ** (attempt + 2)))


# ---------------------------------------------------------------- Steam reviews

def parse_steam_reviews(payload: dict) -> list[dict]:
    """Map a GetAppReviews response to records. The `author` block is dropped entirely."""
    out = []
    for r in payload.get("response", {}).get("reviews", []):
        out.append({
            "id": f"steam:{r['recommendationid']}",
            "source": "steam",
            "ts": int(r["timestamp_created"]),
            "text": clean_text(r.get("review")),
            "score": int(r.get("votes_up", 0)),
            "voted_up": bool(r.get("voted_up")),
        })
    return out


def fetch_steam_reviews(appid: int, start_ts: int, end_ts: int, per_day: int = 15,
                        language: str = "english", limiter: RateLimiter | None = None) -> list[dict]:
    """Even daily sample: up to `per_day` most recent reviews created on each UTC day.

    Uses the documented keyless IUserReviewsService/GetAppReviews endpoint with its
    date_range filter (one request per day, capped at 100 per page).
    """
    limiter = limiter or RateLimiter(1.5)
    seen, out = set(), []
    day = start_ts - start_ts % DAY
    while day < end_ts:
        q = {"appid": appid, "num_per_page": min(per_day, 100), "filter": 1,  # 1 = Recent
             "languages": [language], "purchase_type": 1,  # 1 = All
             "date_range_start": day, "date_range_end": day + DAY - 1}
        payload, _ = http_json(STEAM_REVIEWS + "?input_json=" + urllib.parse.quote(json.dumps(q)), limiter=limiter)
        for rec in parse_steam_reviews(payload):
            if rec["id"] not in seen and day <= rec["ts"] < day + DAY:
                seen.add(rec["id"])
                out.append(rec)
        day += DAY
    return sorted(out, key=lambda r: r["ts"])


# ------------------------------------------------------------------- Steam news

PATCH_RE = re.compile(r"\b(patch|update|hotfix|\d+\.\d+(\.\d+)?)\b", re.I)


def parse_steam_news(payload: dict) -> list[dict]:
    """Official announcements that look like patches/updates -> events."""
    events = []
    for n in payload.get("appnews", {}).get("newsitems", []):
        tags = n.get("tags") or []
        if n.get("feedname") != "steam_community_announcements":
            continue
        if "patchnotes" in tags or PATCH_RE.search(n.get("title", "")):
            events.append({"ts": int(n["date"]), "title": n["title"].strip(), "kind": "patch"})
    return sorted(events, key=lambda e: e["ts"])


def fetch_steam_news(appid: int, count: int = 100, limiter: RateLimiter | None = None) -> list[dict]:
    url = STEAM_NEWS + "?" + urllib.parse.urlencode(
        {"appid": appid, "count": count, "maxlength": 1, "feeds": "steam_community_announcements"})
    payload, _ = http_json(url, limiter=limiter or RateLimiter(1.5))
    return parse_steam_news(payload)


# ----------------------------------------------------------------------- Reddit

REDDIT_TOKEN = "https://www.reddit.com/api/v1/access_token"
REDDIT_API = "https://oauth.reddit.com"


def parse_reddit_listing(payload: dict) -> list[dict]:
    """Map a Listing of t3 (posts) or t1 (comments) to records. `author*` fields are dropped."""
    out = []
    for child in payload.get("data", {}).get("children", []):
        d = child.get("data", {})
        parts = (d.get("title"), d.get("selftext") or d.get("body"))
        text = " ".join(x for x in parts if x and x not in ("[deleted]", "[removed]")).strip()
        if not text:
            continue
        out.append({"id": f"reddit:{d['name']}", "source": "reddit", "ts": int(d["created_utc"]),
                    "text": clean_text(text), "score": int(d.get("score", 0)), "voted_up": None})
    return out


class RedditClient:
    """App-only OAuth (client_credentials). Needs REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET.

    Reddit's free tier allows 100 queries/minute per OAuth client; we stay at one per
    second and also respect the X-Ratelimit-* headers.
    """

    def __init__(self, client_id=None, client_secret=None):
        self.client_id = client_id or os.environ.get("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET")
        if not (self.client_id and self.client_secret):
            raise RuntimeError("set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET (a Reddit 'script' app)")
        self.limiter = RateLimiter(1.0)
        self.token = None

    def _auth(self):
        basic = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        payload, _ = http_json(REDDIT_TOKEN, headers={"Authorization": f"Basic {basic}"},
                               data=b"grant_type=client_credentials", limiter=self.limiter)
        self.token = payload["access_token"]

    def new_posts(self, subreddit: str, pages: int = 5) -> list[dict]:
        if not self.token:
            self._auth()
        out, after = [], None
        for _ in range(pages):
            qs = urllib.parse.urlencode({"limit": 100, "raw_json": 1, **({"after": after} if after else {})})
            payload, headers = http_json(f"{REDDIT_API}/r/{subreddit}/new?{qs}",
                                         headers={"Authorization": f"Bearer {self.token}"}, limiter=self.limiter)
            out += parse_reddit_listing(payload)
            if float(headers.get("X-Ratelimit-Remaining", headers.get("x-ratelimit-remaining", 100))) < 2:
                time.sleep(float(headers.get("X-Ratelimit-Reset", headers.get("x-ratelimit-reset", 60))))
            after = payload.get("data", {}).get("after")
            if not after:
                break
        return out
