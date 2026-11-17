"""Static, dependency-free HTML dashboard (inline SVG) from a pipeline result."""
from __future__ import annotations

import datetime as dt
from html import escape

W, H, PAD = 860, 240, 40


def _date(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).date()


def _sentiment_chart(res) -> str:
    days = sorted(res["series_day"].items())
    if not days:
        return ""
    d0 = dt.date.fromisoformat(days[0][0])
    span = max(1, (dt.date.fromisoformat(days[-1][0]) - d0).days)
    x = lambda d: PAD + (W - 2 * PAD) * (d - d0).days / span
    y = lambda v: H - PAD - (H - 2 * PAD) * (v + 1) / 2
    pts, roll = [], []
    for i, (k, b) in enumerate(days):
        d = dt.date.fromisoformat(k)
        pts.append(f'<circle cx="{x(d):.1f}" cy="{y(b["net"]):.1f}" r="2" class="dot"><title>{k}: net {b["net"]:+.2f}, n={b["n"]}</title></circle>')
        win = days[max(0, i - 6):i + 1]
        n = sum(w["n"] for _, w in win)
        roll.append(f"{x(d):.1f},{y(sum(w['net'] * w['n'] for _, w in win) / n):.1f}")
    ev = "".join(
        f'<line x1="{x(_date(e["ts"])):.1f}" x2="{x(_date(e["ts"])):.1f}" y1="{PAD - 10}" y2="{H - PAD}" class="ev"><title>{escape(e["title"])}</title></line>'
        f'<text x="{x(_date(e["ts"])) + 3:.1f}" y="{PAD - 12}" class="evl">{i + 1}</text>'
        for i, e in enumerate(res["events"]))
    grid = "".join(f'<line x1="{PAD}" x2="{W - PAD}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                   f'<text x="{PAD - 6}" y="{y(v) + 4:.1f}" class="ax" text-anchor="end">{v:+.0f}</text>'
                   for v in (-1, 0, 1))
    labels = "".join(f'<text x="{x(dt.date.fromisoformat(k)):.1f}" y="{H - PAD + 16}" class="ax" text-anchor="middle">{k[5:]}</text>'
                     for k, _ in days[::max(1, len(days) // 8)])
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Daily net sentiment with 7-day rolling mean">'
            f'{grid}{ev}{"".join(pts)}<polyline points="{" ".join(roll)}" class="line"/>{labels}</svg>')


def _topic_heatmap(res) -> str:
    weeks = sorted(res["series_week"].items())
    topics = sorted(res["topics"].items(), key=lambda kv: -kv[1]["size"])
    head = "".join(f"<th>{k[5:]}</th>" for k, _ in weeks)
    rows = []
    for t, info in topics:
        cells = []
        for k, b in weeks:
            n_topic = sum(v["n"] for tt, v in b["topics"].items() if tt != "-1")
            c = b["topics"].get(t, {"n": 0, "net": 0})
            share = c["n"] / n_topic if n_topic else 0
            cells.append(f'<td style="--a:{min(1, share * 3):.2f}" title="{c["n"]} of {n_topic} topical posts, net {c["net"]:+.2f}">'
                         f'{share * 100:.0f}%</td>')
        rows.append(f'<tr><th class="tl">{escape(info["label"])}</th>{"".join(cells)}</tr>')
    return f'<table class="heat"><tr><th>topic \\ week of</th>{head}</tr>{"".join(rows)}</table>'


def _p(p):
    return "" if p is None else "&lt;0.001" if p < 0.001 else f"{p:.3f}"


def _events_table(res) -> str:
    out = []
    for i, e in enumerate(res["events"]):
        s = e["shift"]
        rows = "".join(
            f'<tr class="{t["verdict"].replace(" ", "-")}"><td>{escape(t["name"])}</td>'
            f'<td>{(t["share_before"] or 0) * 100:.0f}%</td><td>{(t["share_after"] or 0) * 100:.0f}%</td>'
            f'<td>{_p(t["p_adj"])}</td><td>{t["verdict"]}</td></tr>'
            for t in s["tests"])
        out.append(f'<details{" open" if i == 0 else ""}><summary><b>{i + 1}. {escape(e["title"])}</b> - {_date(e["ts"])} '
                   f'(n before={s["n_before"]}, after={s["n_after"]}, ±{s["window_days"]} days)</summary>'
                   f'<table><tr><th>measure</th><th>before</th><th>after</th><th>p (Bonferroni)</th><th>verdict</th></tr>{rows}</table></details>')
    return "".join(out) or "<p>No patch announcements inside the sampled range.</p>"


def render(res: dict, eval_summary: dict | None = None) -> str:
    topics = "".join(
        f'<li><b>{escape(v["label"])}</b> ({v["size"]} posts) - {escape(v["description"])}'
        f' <span class="muted">[{escape(", ".join(v["terms"]))}]</span></li>'
        for _, v in sorted(res["topics"].items(), key=lambda kv: -kv[1]["size"]))
    agree = res.get("agreement_with_steam_vote") or {}
    ag = "".join(f"<li>{k}: agrees with the reviewer's own thumbs up/down on {v['agree'] * 100:.1f}% of "
                 f"{v['n_polar']} polar posts ({v['neutral_share'] * 100:.0f}% labelled neutral)</li>"
                 for k, v in agree.items())
    lo, hi = (_date(t) for t in res["range"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(res['game'])} community pulse</title>
<style>
:root{{--bg:#fff;--fg:#1d1f23;--muted:#6b7280;--line:#2563eb;--ev:#dc2626;--grid:#e5e7eb;--heat:37,99,235}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111316;--fg:#e6e7e9;--muted:#9aa1ab;--line:#60a5fa;--ev:#f87171;--grid:#2a2e35;--heat:96,165,250}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:920px;margin:0 auto;padding:16px}}
svg{{width:100%;height:auto}} .grid{{stroke:var(--grid)}} .ax,.evl{{fill:var(--muted);font-size:11px}}
.line{{fill:none;stroke:var(--line);stroke-width:2}} .dot{{fill:var(--muted);opacity:.5}} .ev{{stroke:var(--ev);stroke-dasharray:4 3}}
.muted{{color:var(--muted);font-size:13px}} table{{border-collapse:collapse;font-size:13px;margin:6px 0 14px}}
td,th{{padding:3px 8px;border-bottom:1px solid var(--grid);text-align:right}} .tl,td:first-child{{text-align:left}}
.heat td{{background:rgba(var(--heat),var(--a))}} .wrap{{overflow-x:auto}} .likely-shift td{{font-weight:600}}
.too-few-posts td{{color:var(--muted)}} summary{{cursor:pointer;margin:6px 0}}
</style></head><body>
<h1>{escape(res['game'])}: community pulse</h1>
<p class="muted">{res['n_posts']} sampled Steam reviews, {lo} to {hi}. Sentiment and topic labels come from a local LLM;
red dashed lines are patch announcements from the official Steam news feed.</p>
<h2>Net sentiment</h2><p class="muted">Dots: daily mean of +1 positive / 0 neutral / -1 negative. Line: 7-day rolling mean.</p>
{_sentiment_chart(res)}
<h2>Topic mix by week</h2><p class="muted">Share of topical posts (posts with at least 8 words) in each week.</p>
<div class="wrap">{_topic_heatmap(res)}</div>
<h2>Before vs after each patch</h2>
<p class="muted">Two-proportion z-tests on sampled posts, Bonferroni-adjusted within each event. "too few posts" means a window
had fewer than 30 posts or a cell under 5. A "likely shift" is a change in mix or mood around the date, not proof the patch caused it.</p>
{_events_table(res)}
<h2>Topics</h2><ul>{topics}</ul>
<h2>Sanity check against Steam votes</h2><ul>{ag}</ul>
{"" if not eval_summary else f"<h2>Sentiment accuracy (hand-labelled set)</h2><p>{escape(eval_summary)}</p>"}
</body></html>"""
