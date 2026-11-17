import json
import os
import pathlib
import random
import unittest
from unittest import mock

from community_pulse import analysis as A
from community_pulse import dashboard, evaluate, models, pipeline, sources

FIX = pathlib.Path(__file__).resolve().parent.parent / "fixtures"


def load(name):
    return json.loads((FIX / name).read_text())


class Sources(unittest.TestCase):
    def test_steam_reviews_drop_author(self):
        recs = sources.parse_steam_reviews(load("steam_reviews_page.json"))
        self.assertEqual(len(recs), 5)
        for r in recs:
            self.assertEqual(set(r), {"id", "source", "ts", "text", "score", "voted_up"})
            self.assertTrue(r["id"].startswith("steam:"))
        self.assertNotIn("steamid", json.dumps(recs))

    def test_steam_news_keeps_official_patches_only(self):
        evs = sources.parse_steam_news(load("steam_news.json"))
        titles = [e["title"] for e in evs]
        self.assertIn("Devoid of Liberty: 7.2.3", titles)
        self.assertNotIn("Developer Blog: November 2026", titles)
        self.assertEqual(evs, sorted(evs, key=lambda e: e["ts"]))

    def test_reddit_listing_synthetic(self):
        recs = sources.parse_reddit_listing(load("reddit_listing_synthetic.json"))
        self.assertEqual(len(recs), 5)  # [deleted] post skipped, comment kept
        self.assertNotIn("synthetic_user", json.dumps(recs))
        self.assertTrue(any(r["id"] == "reddit:t1_syn100" for r in recs))

    def test_clean_text_masks_slurs(self):
        self.assertEqual(sources.clean_text("  LIBERTY Faggots!! "), "LIBERTY [slur]!!")
        self.assertEqual(sources.clean_text("nagging fire retardant"), "nagging fire retardant")

    def test_reddit_needs_credentials(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                sources.RedditClient()


class Sentiment(unittest.TestCase):
    def test_lexicon(self):
        self.assertEqual(models.lexicon_sentiment("Great game, love it")["label"], "positive")
        self.assertEqual(models.lexicon_sentiment("Not good, crashes constantly")["label"], "negative")
        self.assertEqual(models.lexicon_sentiment("I bought it on Tuesday")["label"], "neutral")

    def test_llm_sentiment_tolerates_bad_output(self):
        replies = iter([{"results": [{"i": 0, "label": "positive", "confidence": 0.9},
                                     {"i": 1, "label": "furious", "confidence": "x"}]},
                        ValueError("garbled")])

        def chat(_):
            r = next(replies)
            if isinstance(r, Exception):
                raise r
            return r

        out = models.llm_sentiment(["a", "b", "c", "d"], batch=3, chat=chat)
        self.assertEqual(out[0], {"label": "positive", "confidence": 0.9})
        self.assertEqual(out[1], {"label": "neutral", "confidence": 0.0})
        self.assertEqual(out[2]["confidence"], 0.0)  # missing from reply
        self.assertEqual(out[3], {"label": "neutral", "confidence": 0.0})  # garbled batch


class Analysis(unittest.TestCase):
    def test_kmeans_separates_clusters(self):
        rng = random.Random(1)
        a = [[1 + rng.random() * .1, rng.random() * .1] for _ in range(20)]
        b = [[rng.random() * .1, 1 + rng.random() * .1] for _ in range(20)]
        lab = A.kmeans(a + b, 2)
        self.assertEqual(len(set(lab[:20])), 1)
        self.assertEqual(len(set(lab[20:])), 1)
        self.assertNotEqual(lab[0], lab[20])

    def test_two_prop_p(self):
        self.assertAlmostEqual(A.two_prop_p(50, 100, 50, 100), 1.0)
        self.assertLess(A.two_prop_p(10, 100, 40, 100), 0.001)

    def test_fisher_matches_known_value(self):
        # classic tea-tasting table [[3,1],[1,3]]: two-sided p = 0.4857
        self.assertAlmostEqual(A.fisher_p(3, 4, 1, 4), 0.4857, places=3)
        self.assertLess(A.prop_test(4, 105, 32, 105), 0.001)

    def test_merge_events(self):
        evs = A.merge_events([{"ts": 0, "title": "7.2.0"}, {"ts": 86400, "title": "7.2.1"},
                              {"ts": 10 * 86400, "title": "7.3.0"}])
        self.assertEqual([e["title"] for e in evs], ["7.2.0 + 7.2.1", "7.3.0"])

    def _records(self, n_each, neg_before, neg_after):
        t0 = 100 * 86400
        recs = []
        for i in range(n_each):
            recs.append({"ts": t0 - 86400 - i, "sentiment": "negative" if i < neg_before else "positive",
                         "topic": 0, "voted_up": True})
            recs.append({"ts": t0 + 86400 + i, "sentiment": "negative" if i < neg_after else "positive",
                         "topic": 1, "voted_up": True})
        return t0, recs

    def test_event_shift_detects_and_guards(self):
        topics = {0: {"label": "old"}, 1: {"label": "new"}}
        t0, recs = self._records(100, 10, 60)
        s = A.event_shift(recs, t0, topics)
        neg = s["tests"][0]
        self.assertEqual((s["n_before"], s["n_after"]), (100, 100))
        self.assertEqual(neg["verdict"], "likely shift")
        t0, recs = self._records(10, 1, 9)
        self.assertEqual(A.event_shift(recs, t0, topics)["tests"][0]["verdict"], "too few posts")


def fake_chat(prompt):
    if "Posts:" in prompt:
        lines = [l for l in prompt.split("Posts:\n")[1].splitlines() if l.strip()]
        return {"results": [{"i": int(l.split(":")[0]),
                             "label": "negative" if "crash" in l else "positive", "confidence": .8} for l in lines]}
    return {"label": "Crashes" if "crash" in prompt else "Fun", "description": "test"}


def fake_embed(texts):
    return [[1.0, 0.0] if "crash" in t else [0.0, 1.0] for t in texts]


class EndToEnd(unittest.TestCase):
    def test_pipeline_and_dashboard(self):
        t0 = 1_790_000_000
        posts = []
        for d in range(20):
            for j in range(10):
                bad = j < (6 if d >= 10 else 1)
                txt = ("the game keeps crashing after the patch every single time" if bad
                       else "really fun co-op with friends every evening this week")
                posts.append({"id": f"x{d}-{j}", "source": "steam", "ts": t0 + d * 86400 + j, "text": txt,
                              "score": 0, "voted_up": not bad})
        res = pipeline.run(posts, [{"ts": t0 + 10 * 86400, "title": "Patch 1.1", "kind": "patch"}], "Test Game",
                           k=2, chat=fake_chat, embed=fake_embed, progress=lambda *_: None)
        self.assertEqual(res["n_posts"], 200)
        self.assertEqual(sorted(v["label"] for v in res["topics"].values()), ["Crashes", "Fun"])
        neg = res["events"][0]["shift"]["tests"][0]
        self.assertEqual(neg["verdict"], "likely shift")
        self.assertEqual(res["agreement_with_steam_vote"]["sentiment"]["agree"], 1.0)
        html = dashboard.render(res, "summary")
        self.assertIn("<svg", html)
        self.assertIn("Patch 1.1", html)


class Eval(unittest.TestCase):
    def test_score_and_mcnemar(self):
        s = evaluate.score(["positive", "negative", "neutral", "positive"],
                           ["positive", "negative", "positive", "positive"])
        self.assertEqual(s["accuracy"], 0.75)
        self.assertEqual(s["confusion_gold_x_pred"]["neutral"]["positive"], 1)
        self.assertEqual(evaluate.mcnemar_exact(0, 0), 1.0)
        self.assertLess(evaluate.mcnemar_exact(0, 10), 0.01)

    def test_labels_file_is_valid(self):
        path = FIX.parent / "eval" / "labels.jsonl"
        items = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        self.assertGreaterEqual(len(items), 100)
        self.assertTrue(all(it["label"] in models.LABELS and it["text"] for it in items))
        self.assertEqual(len({it["id"] for it in items}), len(items))


if __name__ == "__main__":
    unittest.main()


class Parsing(unittest.TestCase):
    def test_parse_json_tolerates_thoughts_and_fences(self):
        self.assertEqual(models.parse_json('<thought>{"x": 0} hmm</thought>```json\n{"a": 1}\n```'), {"a": 1})
        with self.assertRaises(ValueError):
            models.parse_json("no json here")


class Demo(unittest.TestCase):
    def test_replay_demo_reproduces_committed_example(self):
        import tempfile
        from community_pulse import __main__ as cli
        saved = (dict(models.REPLAY), models.PROVIDER, models.EMBED_PROVIDER, models.CHAT_MODEL, models.EMBED_MODEL)
        try:
            with tempfile.TemporaryDirectory() as d, mock.patch("builtins.print"), \
                    mock.patch.dict(os.environ, {"GEMINI_API_KEY": "", "CP_PROVIDER": ""}):
                cli.main(["demo", "--out", d])
                got = json.loads(pathlib.Path(d, "result.json").read_text())
        finally:
            models.REPLAY.clear(); models.REPLAY.update(saved[0])
            models.PROVIDER, models.EMBED_PROVIDER, models.CHAT_MODEL, models.EMBED_MODEL = saved[1:]
        want = json.loads((FIX.parent / "examples/helldivers2/result.json").read_text())
        self.assertEqual(got["models"]["usage"]["chat_calls"] + got["models"]["usage"]["embed_calls"], 0)
        for k in ("topics", "events", "series_week", "agreement_with_steam_vote"):
            self.assertEqual(got[k], want[k], k)
