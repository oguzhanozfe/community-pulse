"""CLI: python -m community_pulse {demo,fetch-steam,fetch-reddit,run,render,eval} ..."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

from . import analysis, dashboard, evaluate, models, pipeline, sources


ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples/helldivers2"


def _ts(s: str) -> int:
    return int(dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="community_pulse")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("demo", help="bundled Helldivers 2 run; no key needed (cloud key -> Ollama with --live -> recorded replay)")
    m.add_argument("--out", default="out/demo")
    m.add_argument("--live", action="store_true", help="use local Ollama when no cloud key is set (slow)")

    f = sub.add_parser("fetch-steam", help="sample Steam reviews + patch dates (no key needed)")
    f.add_argument("--appid", type=int, required=True)
    f.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    f.add_argument("--end", required=True, help="YYYY-MM-DD (UTC), exclusive")
    f.add_argument("--per-day", type=int, default=15)
    f.add_argument("--out", required=True)

    r = sub.add_parser("fetch-reddit", help="newest posts via Reddit's OAuth API (needs env credentials)")
    r.add_argument("--subreddit", required=True)
    r.add_argument("--pages", type=int, default=5)
    r.add_argument("--out", required=True)

    p = sub.add_parser("run", help="analyse a fetched file and write result.json + dashboard.html")
    p.add_argument("--input", required=True)
    p.add_argument("--game", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=8, help="number of topic clusters")
    p.add_argument("--window", type=int, default=7, help="days before/after each patch")
    p.add_argument("--eval-results", help="eval/results.json to quote on the dashboard")

    d = sub.add_parser("render", help="re-render dashboard.html from an existing result.json")
    d.add_argument("--result", required=True)
    d.add_argument("--eval-results")

    e = sub.add_parser("eval", help="score LLM and lexicon sentiment on the hand-labelled set")
    e.add_argument("--labels", default="eval/labels.jsonl")
    e.add_argument("--out", default="eval/results.json")

    a = ap.parse_args(argv)
    if a.cmd == "demo":
        print(models.resolve_backend(EXAMPLE / "replay.json.gz", live=a.live))
        a = ap.parse_args(["run", "--input", str(EXAMPLE / "input.json"), "--game", "Helldivers 2",
                           "--out", a.out, "--eval-results", str(ROOT / "eval/results.json")])
    if a.cmd == "fetch-steam":
        data = {"source": "steam", "appid": a.appid,
                "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "events": sources.fetch_steam_news(a.appid),
                "posts": sources.fetch_steam_reviews(a.appid, _ts(a.start), _ts(a.end), a.per_day)}
        _write(a.out, data)
        print(f"{len(data['posts'])} reviews, {len(data['events'])} patch announcements -> {a.out}")
    elif a.cmd == "fetch-reddit":
        data = {"source": "reddit", "subreddit": a.subreddit, "events": [],
                "posts": sources.RedditClient().new_posts(a.subreddit, a.pages)}
        _write(a.out, data)
        print(f"{len(data['posts'])} posts -> {a.out}")
    elif a.cmd == "run":
        data = json.loads(pathlib.Path(a.input).read_text())
        res = pipeline.run(data["posts"], data.get("events", []), a.game, a.k, a.window)
        res["input"] = {k: v for k, v in data.items() if k not in ("posts", "events")}
        res["models"] = models.model_info()
        summary = evaluate.summary_line(json.loads(pathlib.Path(a.eval_results).read_text())) if a.eval_results else None
        out = pathlib.Path(a.out)
        _write(out / "result.json", res)
        (out / "dashboard.html").write_text(dashboard.render(res, summary))
        print(f"-> {out}/result.json, {out}/dashboard.html")
    elif a.cmd == "render":
        res = json.loads(pathlib.Path(a.result).read_text())
        # recompute the before/after tests from saved posts (no model calls)
        topics = {int(t): v for t, v in res["topics"].items()}
        for e in res["events"]:
            e["shift"] = analysis.event_shift(res["posts"], e["ts"], topics, e["shift"]["window_days"])
        _write(a.result, res)
        summary = evaluate.summary_line(json.loads(pathlib.Path(a.eval_results).read_text())) if a.eval_results else None
        out = pathlib.Path(a.result).with_name("dashboard.html")
        out.write_text(dashboard.render(res, summary))
        print(f"-> {out}")
    elif a.cmd == "eval":
        res = evaluate.run(a.labels)
        _write(a.out, res)
        print(json.dumps({k: v for k, v in res.items() if k != "items"}, indent=1))


def _write(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
