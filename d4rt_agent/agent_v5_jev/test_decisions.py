import json

import pytest

from d4rt_agent.agent_v5_jev import decisions as D


def _fake_response(questions, pick="A"):
    answers = []
    for q in questions:
        values = [c["value"] for c in q["choices"]]
        answers.append({"type": "choice", "name": q["name"], "choice": pick,
                        "probabilities": [{"value": v, "probability": 1.0 if v == pick else 0.0}
                                          for v in values], "confidence": 1.0})
    return {"model": D.DEFAULT_MODEL, "answers": answers, "usage": {"input_tokens": 12}}


QS = [D.choice_question("kind", "Which kind?", [("A", "camera moves"), ("B", "subject moves")])]


def test_request_body_and_key_are_stable():
    body = D.request_body("text", QS)
    assert body["model"] == "gpt-6-luna" and body["questions"][0]["choices"][0] == {
        "value": "A", "description": "camera moves"}
    assert D.request_key(body) == D.request_key(json.loads(json.dumps(body)))
    assert D.request_key(body) != D.request_key(D.request_body("other text", QS))


def test_duplicate_names_and_values_rejected():
    with pytest.raises(ValueError):
        D.choice_question("q", "x", [("A", "a"), ("A", "b")])
    with pytest.raises(ValueError):
        D.request_body("t", QS + QS)


def test_live_call_is_cached(tmp_path):
    calls = []

    def poster(body):
        calls.append(body)
        return _fake_response(body["questions"], "B")

    first = D.decide("t", QS, tmp_path, poster=poster)
    second = D.decide("t", QS, tmp_path, poster=poster)
    assert first == second == {"kind": {"choice": "B", "probs": {"A": 0.0, "B": 1.0}, "confidence": 1.0}}
    assert len(calls) == 1
    assert json.loads((tmp_path / "usage.jsonl").read_text().splitlines()[0])["input_tokens"] == 12


def test_collect_then_send_then_cached(tmp_path):
    with pytest.raises(D.NeedRequest):
        D.decide("t", QS, tmp_path, mode="collect")
    assert len(list((tmp_path / "pending").glob("*.json"))) == 1
    sent, failed = D.send_pending(tmp_path, poster=lambda body: _fake_response(body["questions"]))
    assert (sent, failed) == (1, [])
    assert not list((tmp_path / "pending").glob("*.json"))
    assert D.decide("t", QS, tmp_path, mode="cached")["kind"]["choice"] == "A"
    with pytest.raises(KeyError):
        D.decide("other", QS, tmp_path, mode="cached")


def test_refusal_raises():
    response = {"answers": [{"type": "refusal", "name": "kind"}]}
    with pytest.raises(D.Refused):
        D.parse_answers(response, QS)


def test_send_pending_reports_failures(tmp_path):
    with pytest.raises(D.NeedRequest):
        D.decide("t", QS, tmp_path, mode="collect")

    def boom(body):
        raise RuntimeError("HTTP 403")

    sent, failed = D.send_pending(tmp_path, poster=boom)
    assert sent == 0 and "HTTP 403" in failed[0]
    assert len(list((tmp_path / "pending").glob("*.json"))) == 1


def test_same_request_from_many_threads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    def poster(body):
        return _fake_response(body["questions"])

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: D.decide("same text", QS, tmp_path, poster=poster), range(64)))
    assert all(r["kind"]["choice"] == "A" for r in results)
    assert not list((tmp_path / "responses").glob("*.tmp*"))


def test_refusal_is_retried_and_never_cached(tmp_path):
    calls = []

    def flaky(body):
        calls.append(1)
        if len(calls) == 1:
            return {"answers": [{"type": "refusal", "name": "kind"}], "usage": {"input_tokens": 1}}
        return _fake_response(body["questions"])

    assert D.decide("t", QS, tmp_path, poster=flaky)["kind"]["choice"] == "A" and len(calls) == 2

    def always(body):
        return {"answers": [{"type": "refusal", "name": "kind"}], "usage": {"input_tokens": 1}}

    with pytest.raises(D.Refused):
        D.decide("other", QS, tmp_path, poster=always)
    assert len(list((tmp_path / "responses").glob("*.json"))) == 1


def test_single_choice_questions_are_answered_locally(tmp_path):
    sent = []

    def poster(body):
        sent.append(body)
        return _fake_response(body["questions"], "B")

    one = D.choice_question("only", "x", [("A", "the only way")])
    out = D.decide("t", [one], tmp_path, poster=poster)
    assert out["only"]["choice"] == "A" and out["only"]["forced"] and not sent
    mixed = D.decide("t", [one] + QS, tmp_path, poster=poster)
    assert mixed["only"]["choice"] == "A" and mixed["kind"]["choice"] == "B"
    assert [q["name"] for q in sent[0]["questions"]] == ["kind"]
