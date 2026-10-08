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
