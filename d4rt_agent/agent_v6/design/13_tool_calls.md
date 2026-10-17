# 13 — Calling tools reliably

Theme 4 of the discussion. It covers tracker points P1, P4 and P10. Status: **settled** on
2026-10-17 (decision log #53, #54, #59–63, with #48 and #55).

## What fails today

"Tool failure" covered three different things in V5 ([03](03_evidence_so_far.md) §6):

- **Broken model output.** Invalid JSON: rare in legacy V5 except at the answer turn; frequent with
  the compositional schemas.
- **Refused calls.** The host rejected arguments that meant something sensible: entity IDs,
  generator-frame windows, `cam` not registered, inspection handles. About 470 per compositional run;
  on compositional DSI, 100 of 252 questions had one, and they scored 32%.
- **Unavailable measurements.** Perception failed: SAM3 found nothing, Orient-Anything found no
  front, too few usable frames.

V6 handles each layer differently.

## Layer 1 — design the causes away

Most refusals came from forms the model could not produce reliably. V6 removes those forms:

- entities are named in words, never IDs (#48);
- times are seconds or the labels of frames shown, never generator-frame integers (P4; [10](10_tool_reports.md), #41);
- the camera is always an implicit reference, never an entity to register (#20);
- no inspection or visual handles.

## Layer 2 — normaliser, then doctor (#59–62)

```
agent's call ──► normaliser (host, no model)                                         #59
                   repairs JSON (code fences, quotes, trailing text); fuzzy-matches argument
                   names (frame → frame_of_reference) and values ("subj" → subject); maps
                   entity names to known ones ("the runner" → runner, "car 2"); fills defaults
              ──► still invalid? doctor (local Qwen3-VL-8B, separate short text-only context) #60
                   sees: the raw call, the plan, the last 2 turns, that tool's full schema
                         and two examples
                   returns: a corrected call (validated by the host, then run)
                            or 2–3 lines telling the agent what to fix
              ──► the agent always sees one line: "corrected: frame → frame_of_reference=subject"
```

**Doctor limits (#61).** The doctor may fill missing values and replace invalid ones. It never
changes a value the agent filled validly. If a fix would need that, it sends the question back to
the agent instead. One doctor attempt per failed call.

**Budget (#62).** A corrected call counts once towards the 14 tool calls. Doctor runs are free, capped
at 5 per question.

**The final answer gets the same treatment (#63).** If the answer is malformed, the host first
extracts the letter deterministically when the text clearly states one. Otherwise the doctor tries.
Neither ever invents a letter; a question with no recoverable answer stays unanswered. V5's VLM4D
run lost 24 questions this way.

## Layer 3 — when perception fails

These are host policies, not doctor work:

- **Grounding (#53).** After the first SAM3 miss: "No object found with the name <name>. Suggest an
  alternative name: singular, short, descriptive words." After the second: "Still not found. This is
  not proof that it is absent: check with `get_frames` or `ask_video`."
- **Facing (#54, #58).** When Orient-Anything finds no front, a VLM is shown a crop of the subject and
  picks one of eight camera-view labels with greedy decoding. It is queried 3 times, at the start,
  middle and end of the clip or at given times. Details are in [11](11_toolset.md).
- **Motion survives (#56).** Without facing, `track_entity` still reports motion in the world and
  camera frames; only facing-dependent lines are marked unavailable.

## Implementation notes

- **Order of checks.** Parse → normalise → validate against the schema → doctor (if still invalid) →
  validate the doctor's call → run. Every step is logged in the trace with before and after values,
  so evaluation can count normaliser fixes, doctor fixes and calls sent back to the agent.
- **What the normaliser may do.** Fuzzy matching uses edit distance against the schema's names and
  enums only. A match must be unique and close; ties go to the doctor. Entity names match against
  the entities from planning, the inventory, and instance labels.
- **The doctor's prompt.** Short and fixed: the tool's schema with its one-line when / when-not, two
  valid example calls, the failed call and the error. Its output is either one JSON call or text
  starting "To the agent:".
- **Measuring it.** Report how many calls needed the normaliser or the doctor, how many were sent
  back, and accuracy on questions with any correction, against V5's refused-call rates in 03 §6.
