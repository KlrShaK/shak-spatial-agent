from d4rt_agent.agent_v5_jev import caption as C

TEXT = """Entities:
- man in red jacket | first seen 0.0 s | last seen 2.5 s | walks left
- white dog | first seen 0.5 s | last seen 2.5 s | runs toward the camera
- white dog | first seen 0.5 s | last seen 2.5 s | runs toward the camera

Timeline:
- 0.0-1.0 s: the man walks left
- 1.0-2.5 s: the dog runs toward the camera
"""


def test_entities_stop_at_the_timeline_and_are_unique():
    ents = C.parse_entities(TEXT)
    assert [e["name"] for e in ents] == ["man in red jacket", "white dog"]
    assert ents[0]["first_s"] == 0.0 and ents[0]["last_s"] == 2.5 and ents[0]["behaviour"] == "walks left"


def test_spans():
    assert C.timeline_spans(TEXT) == [(0.0, 1.0), (1.0, 2.5)]


def test_missing_entities_header_gives_nothing():
    assert C.parse_entities("- 0-1 s: a dog walks") == []
    assert C.parse_entities("") == []


def test_single_timestamps_become_spans():
    text = "Timeline:\n- 0.00 s: a dog stands\n- 0.50 s: it runs\n- 1.00 s: it stops\n"
    assert C.timeline_spans(text) == [(0.0, 0.5), (0.5, 1.0)]


class _Clip:
    answer_frames = list(range(16))
    slug = "x"


def test_caption_clip_assembles_the_two_steps():
    calls = []

    def fake(qwen, msgs, tokens, rp):
        calls.append((len([p for p in msgs[0]["content"] if p["type"] == "image"]), tokens, rp))
        return ("- dog | first seen 0 s | last seen 2 s | runs" if len(calls) == 1 else "- 0-2 s: the dog runs"), 1, 1

    class Clip(_Clip):
        def label(self, i): return f"Frame {i}"
        def pil(self, i, side): return object()

    text = C.caption_clip(None, Clip(), gen=fake)
    assert calls == [(8, 250, 1.15), (8, 250, 1.15)]
    assert [e["name"] for e in C.parse_entities(text)] == ["dog"] and C.timeline_spans(text) == [(0.0, 2.0)]


def test_span_formats_the_small_model_writes():
    text = "Timeline:\n-0.00 to 0.33 s: a\n-0.33 to 0.70 s: b\n- 1.0 - 2.0 s: c\n"
    assert C.timeline_spans(text) == [(0.0, 0.33), (0.33, 0.7), (1.0, 2.0)]


def test_load_caption_reparses_the_text(tmp_path):
    import json

    (tmp_path / "x.json").write_text(json.dumps({"slug": "x", "text": "Entities:\n- dog | first seen 0 s | last seen 1 s | runs\n\nTimeline:\n-0 to 1 s: runs",
                                                  "entities": [], "spans": []}))
    rec = C.load_caption(tmp_path, "x")
    assert [e["name"] for e in rec["entities"]] == ["dog"] and rec["spans"] == [(0.0, 1.0)]
