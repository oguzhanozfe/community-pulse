"""Model calls (Gemini by default; DeepSeek or Ollama optional) and the lexicon baseline."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import pathlib
import random
import re
import time
import urllib.error
import urllib.request

# Every backend is reached through its OpenAI-compatible API, so switching is one env var.
PROVIDERS = {
    "gemini": {"base": "https://generativelanguage.googleapis.com/v1beta/openai", "key": "GEMINI_API_KEY",
               "chat": "gemini-3.1-flash-lite", "embed": "gemini-embedding-001", "interval": 4.5},
    "deepseek": {"base": "https://api.deepseek.com/v1", "key": "DEEPSEEK_API_KEY",
                 "chat": "deepseek-chat", "embed": None, "interval": 0.5},
    "ollama": {"base": os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434") + "/v1", "key": None,
               "chat": "qwen3.5:9b", "embed": "qwen3-embedding:0.6b", "interval": 0.0},
}
PROVIDER = os.environ.get("CP_PROVIDER", "gemini")              # chat: gemini | deepseek | ollama
EMBED_PROVIDER = os.environ.get("CP_EMBED_PROVIDER", "gemini")  # embeddings: gemini | ollama
CHAT_MODEL = os.environ.get("CP_CHAT_MODEL") or PROVIDERS[PROVIDER]["chat"]
EMBED_MODEL = os.environ.get("CP_EMBED_MODEL") or PROVIDERS[EMBED_PROVIDER]["embed"]
LABELS = ("positive", "negative", "neutral")
USAGE = {"cache_hits": 0, "chat_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "embed_calls": 0}
_last_call = {}


def _post(provider: str, path: str, body: dict, timeout=120, retries=4) -> dict:
    """POST JSON with per-provider pacing; backs off on 429/5xx/timeouts. Keys come from env only."""
    cfg = PROVIDERS[provider]
    headers = {"Content-Type": "application/json"}
    if cfg["key"]:
        key = os.environ.get(cfg["key"])
        if not key:
            raise RuntimeError(f"set {cfg['key']} or choose another backend with CP_PROVIDER / CP_EMBED_PROVIDER")
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(cfg["base"] + path, data=json.dumps(body).encode(), headers=headers)
    for attempt in range(retries + 1):
        wait = _last_call.get(provider, 0) + float(os.environ.get("CP_MIN_INTERVAL", cfg["interval"])) - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[provider] = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except (TimeoutError, ConnectionError, http.client.HTTPException, urllib.error.URLError) as e:
            code = getattr(e, "code", None)
            body = e.read().decode("utf-8", "replace") if isinstance(e, urllib.error.HTTPError) else ""
            if code == 429 and "PerDay" in body:
                raise RuntimeError(f"{provider}: daily quota exhausted for this model; try CP_CHAT_MODEL") from e
            if attempt == retries or (code is not None and code != 429 and code < 500):
                raise
            time.sleep(min(90, 5 * 2 ** attempt) * (0.5 + random.random()))  # jitter: shared quotas


def model_info() -> dict:
    return {"chat_provider": PROVIDER, "chat": CHAT_MODEL, "embed_provider": EMBED_PROVIDER,
            "embed": EMBED_MODEL, "usage": dict(USAGE)}


CACHE = pathlib.Path(os.environ.get("CP_CACHE", ".cache"))


def _cached(kind: str, model: str, payload, call):
    """Disk cache keyed by model + input hash, so reruns cost no quota."""
    key = hashlib.sha256(json.dumps([kind, model, payload], sort_keys=True).encode()).hexdigest()
    path = CACHE / kind / f"{key}.json"
    if path.exists():
        USAGE["cache_hits"] += 1
        return json.loads(path.read_text())
    out = call()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def parse_json(text: str) -> dict:
    """Model text -> JSON object; tolerates <thought> blocks and code fences."""
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start:end + 1])


def chat_json(prompt: str) -> dict:
    """One deterministic chat turn returning a JSON object."""
    def call():
        body = {"model": CHAT_MODEL, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
        if not CHAT_MODEL.startswith("gemma"):  # Gemma on the Gemini API rejects JSON mode
            body["response_format"] = {"type": "json_object"}
        out = _post(PROVIDER, "/chat/completions", body, timeout=300)
        USAGE["chat_calls"] += 1
        for k in ("prompt_tokens", "completion_tokens"):
            USAGE[k] += (out.get("usage") or {}).get(k, 0) or 0
        return out["choices"][0]["message"]["content"]
    return parse_json(_cached("chat", f"{PROVIDER}/{CHAT_MODEL}", prompt, call))


def embed(texts: list[str], dims: int = 256, batch: int = 100) -> list[list[float]]:
    vecs = []
    for i in range(0, len(texts), batch):
        chunk = [t[:1500] for t in texts[i:i + batch]]

        def call():
            out = _post(EMBED_PROVIDER, "/embeddings", {"model": EMBED_MODEL, "dimensions": dims, "input": chunk})
            USAGE["embed_calls"] += 1
            return [d["embedding"] for _, d in sorted(enumerate(out["data"]), key=lambda x: x[1].get("index", x[0]))]
        vecs += _cached("embed", f"{EMBED_PROVIDER}/{EMBED_MODEL}/{dims}", chunk, call)
    return vecs


# ------------------------------------------------------------ LLM sentiment

SENTIMENT_PROMPT = """You rate the sentiment of player posts about the video game "{game}".
For each numbered post decide the writer's overall feeling about the game:
- positive: mostly praise or recommends it
- negative: mostly complaints or does not recommend it
- neutral: mixed with no clear lean, a question, a joke, or no opinion
Give a confidence from 0 to 1 that your label is right.
Return JSON only: {{"results": [{{"i": <number>, "label": "positive|negative|neutral", "confidence": <0-1>}}]}}

Posts:
{posts}"""


def llm_sentiment(texts: list[str], game: str = "this game", batch: int = 150, chat=None) -> list[dict]:
    """[{label, confidence}] per text. Items the model skips or garbles fall back to neutral/0.0."""
    chat = chat or chat_json
    out = []
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        posts = "\n".join(f"{i}: {' '.join(t.split())[:600]}" for i, t in enumerate(chunk))
        try:
            got = {int(r["i"]): r for r in chat(SENTIMENT_PROMPT.format(game=game, posts=posts)).get("results", [])}
        except (ValueError, KeyError, TypeError):
            got = {}
        for i in range(len(chunk)):
            r = got.get(i, {})
            label = r.get("label") if r.get("label") in LABELS else "neutral"
            try:
                conf = max(0.0, min(1.0, float(r.get("confidence", 0.0))))
            except (TypeError, ValueError):
                conf = 0.0
            out.append({"label": label, "confidence": conf if i in got else 0.0})
    return out


# -------------------------------------------------------- lexicon baseline

POS = set("""good great love loved loving awesome amazing excellent fun enjoy enjoyed enjoying best fantastic
beautiful masterpiece recommend recommended perfect addictive addicting solid brilliant incredible
wonderful favorite favourite gem worth polished smooth improved improvement better nice cool epic
satisfying relaxing chill immersive gorgeous peak goat banger fixed outstanding superb blast""".split())
NEG = set("""bad terrible awful hate hated boring broken bug bugs buggy crash crashes crashing refund
refunded worst trash garbage waste disappointing disappointed disappointment unplayable laggy lag
cheaters cheater cheating greedy scam ruined ruin ruins annoying frustrating frustrated dead stutter
stuttering grind grindy grinding overpriced mediocre meh poor worse unfinished nerf nerfed nerfs
tedious toxic glitch glitches glitchy kick kicked ban banned shame pathetic sucks""".split())
NEGATORS = {"not", "no", "never", "dont", "don't", "isnt", "isn't", "wasnt", "wasn't", "cant", "can't",
            "doesnt", "doesn't", "nothing", "hardly", "without"}
WORD = re.compile(r"[a-z']+")


def lexicon_sentiment(text: str) -> dict:
    """Word-count baseline; a negator flips the next sentiment word within 3 words. Ties -> neutral."""
    score, negate = 0, 0
    for w in WORD.findall(text.lower()):
        s = 1 if w in POS else -1 if w in NEG else 0
        if w in NEGATORS:
            negate = 3  # flips the next sentiment word within 3 words
        elif s:
            score += -s if negate else s
            negate = 0
        else:
            negate = max(0, negate - 1)
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return {"label": label, "confidence": min(1.0, abs(score) / 3)}
