"""What a choice-only decision model may pick from: entity phrases and time windows.

The Decisions API returns one of the options it is given; it cannot write a phrase. So the host
proposes phrases (the question's own noun phrases, and the entities the caption names) and the
model picks. Every phrase is then checked by ``kinds.validate_slot``, the same rule AgentV5 applies
to Qwen: an entity must use a noun from the question.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from d4rt_agent.agent_v5.kinds import Slot, is_camera, validate_slot
from d4rt_agent.agent_v5.program import phrase_ladder, singular

MAX_WORDS = 4
MAX_CANDIDATES = 10

_DETERMINERS = {"the", "a", "an", "two", "three", "four", "five", "each", "this", "that", "these", "those",
                "both", "all", "any", "one", "every"}
# a noun phrase ends before any of these
_STOP = set("""
is are was were be been being do does did has have had will would can could should may might
how what which who whom whose when where why whether while during after before until since
to of in on at by for with without from into onto over under above below behind beside near next
between among through across along around toward towards away up down out off than then as and or but
if so not no yes only just also more most less least very much many few some such own same other
moves move moving moved changes change changing changed stays stay remains remain get gets getting
become becomes appear appears appearing happens happen start starts begin begins end ends
walks walk runs run turns turn goes go comes come looks look seems seem shows show
first last initially finally eventually then
they it he she them their its his her we you i our us
still throughout entire overall approximate approximately average primary
increase increases decrease decreases rides ride evolves evolve develop develops unfolds unfold progresses progress
""".split())
# nouns that name a measured quantity or the footage, never the entity
_ABSTRACT = set("""
position positions distance distances size sizes speed speeds direction directions motion motions movement
movements orientation angle angles height order relationship relation arrangement number trajectory path
paths location locations appearance video clip frame frames scene scenes view views footage camera
perspective viewpoint shot shots beginning start end middle half quarter second seconds time times moment
moments period part parts side sides way ways type kind sequence rate change changes level amount
spatial relative composition course activity
""".split())


def _tokens(text: str) -> list[str]:
    text = re.sub(r"['’]s\b", "", text)
    return re.findall(r"[a-z]+(?:-[a-z]+)*|[^\sa-z]", text.lower())


def question_phrases(question: str) -> list[str]:
    """Noun phrases of the question: runs of content words after a determiner, each with the
    simpler rungs of ``phrase_ladder`` (clause cut, head noun), plus the noun after "how many"
    or "number of".

    ``"how does the red car moving left change relative to the two dogs?"`` gives ``red car moving
    left``, ``red car``, ``car``, ``dogs``, ``dog``. Abstract nouns (position, distance, ...) never
    start or form a phrase."""

    tokens = _tokens(question)
    found: list[str] = []
    i = 0
    while i < len(tokens):
        counted = tokens[i - 2:i] == ["how", "many"] or tokens[i - 2:i] == ["number", "of"]
        if tokens[i] in _DETERMINERS or counted:
            j = i if counted else i + 1
            run: list[str] = []
            while j < len(tokens) and re.fullmatch(r"[a-z]+(?:-[a-z]+)*", tokens[j]) \
                    and tokens[j] not in _STOP and tokens[j] not in _DETERMINERS and len(run) < MAX_WORDS \
                    and not (len(tokens[j]) > 4 and tokens[j].endswith("ly")):
                run.append(tokens[j])
                j += 1
            run = _trim(run)
            if run:
                found.extend(phrase_ladder(" ".join(run)))
                found.extend(" ".join(run[:n]) for n in (1, 2) if n < len(run))   # "snowboarder jumps" -> "snowboarder"
            i = max(j, i + 1)
        else:
            i += 1
    return _unique(found)


def _trim(run: list[str]) -> list[str]:
    while run and run[0] in _ABSTRACT:
        run = run[1:]
    while run and run[-1] in _ABSTRACT:
        run = run[:-1]
    return [w for w in run] if run and run[-1] not in _ABSTRACT else []


def _unique(phrases: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for phrase in phrases:
        key = " ".join(singular(w) for w in phrase.lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(phrase)
    return out


def head_noun(phrase: str) -> str:
    """The head of a phrase: its last word after cutting at the first preposition or participle."""

    words = phrase_ladder(phrase)[-1].split()
    return singular(words[-1]) if words else ""


def entity_candidates(slot: Slot, question: str, caption_entities: Sequence[str] = ()) -> list[str]:
    """Phrases the decision model may choose for an entity/category slot, best first.

    Caption entities come first when they pass the question-words rule (they carry the colour or
    position detail that tells similar things apart); the question's own phrases follow. An
    ``entity_or_camera`` slot also offers ``camera``. A slot that allows the implicit main subject
    ("the main object") may use any caption entity."""

    out: list[str] = []
    if slot.type == "entity_or_camera":
        out.append("camera")
    pool = [*caption_entities, *question_phrases(question)]
    for phrase in pool:
        phrase = " ".join(phrase.split())
        if not phrase or is_camera(phrase):
            continue
        try:
            value = validate_slot(slot, phrase, question)
        except ValueError:
            continue
        if value not in out:
            out.append(value)
    return _dedupe_by_head(out)[:MAX_CANDIDATES]


def _dedupe_by_head(phrases: list[str]) -> list[str]:
    """Keep the first of phrases that are the same string after singularising."""

    seen: set[str] = set()
    out = []
    for phrase in phrases:
        key = " ".join(singular(w) for w in phrase.lower().split())
        if key not in seen:
            seen.add(key)
            out.append(phrase)
    return out
