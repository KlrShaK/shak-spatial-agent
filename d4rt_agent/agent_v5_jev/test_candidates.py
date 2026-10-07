from d4rt_agent.agent_v5.kinds import KIND_BY_NAME
from d4rt_agent.agent_v5_jev import candidates as C

SUBJECT = KIND_BY_NAME["subject_direction"].slots[0]
MAIN = KIND_BY_NAME["cam_subject_range"].slots[0]
REFERENCE = KIND_BY_NAME["pair_relation"].slots[1]
CATEGORY = KIND_BY_NAME["instance_count"].slots[0]


def test_named_subject_comes_from_the_question():
    q = "How does the position of the red car change relative to the two dogs?"
    cands = C.entity_candidates(SUBJECT, q)
    assert "red car" in cands and "car" in cands and "dogs" in cands
    assert not any(c in cands for c in ("position", "distance"))


def test_participle_and_verb_tails_are_trimmed():
    cands = C.entity_candidates(SUBJECT, "Does the snowboarder jumping onto the rail move left?")
    assert "snowboarder" in cands


def test_possessive_and_prefix():
    assert "girl" in C.entity_candidates(SUBJECT, "How does the girl's interaction with the kitten evolve?")


def test_reference_slot_offers_the_camera():
    cands = C.entity_candidates(REFERENCE, "How does the gap between the man and the camera change?")
    assert cands[0] == "camera" and "man" in cands


def test_category_after_how_many():
    assert "person" in C.entity_candidates(CATEGORY, "How many people are in the video?")


def test_main_subject_needs_the_caption_and_gets_it():
    q = "How does the distance between the camera and the main subject change?"
    assert C.entity_candidates(MAIN, q) == [] or "man in blue" not in C.entity_candidates(MAIN, q)
    cands = C.entity_candidates(MAIN, q, caption_entities=["man in blue jacket", "white dog"])
    assert "man in blue jacket" in cands and "white dog" in cands


def test_caption_entities_must_use_the_questions_nouns_when_named():
    q = "Where is the dog relative to the fence?"
    cands = C.entity_candidates(SUBJECT, q, caption_entities=["brown dog", "tree"])
    assert "brown dog" in cands and "tree" not in cands


def test_candidates_are_unique_and_capped():
    cands = C.entity_candidates(SUBJECT, "Where is the dog relative to the dogs near the dog on the dog?")
    assert len(cands) == len(set(cands)) <= C.MAX_CANDIDATES
