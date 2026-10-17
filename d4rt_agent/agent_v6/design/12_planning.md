# 12 — Planning before measuring

Theme 3 of the discussion. It covers tracker points P5, P6, P8, P12 and T11. Status: **inputs,
prompt, plan format, budget and orchestration agreed** on 2026-10-17 (decision log #7, #28–30,
#32–38, #44–53, #57). Settled.

## The flow

```
captioning subagent   sees the clip; no question, no planning context
                      → caption of about 50 words: entities, distinguishing attributes, setting,
                        first-person or not; no motion directions
main agent            gets system prompt + caption + question + options + 8 keyframes
  1. Which entities are relevant?
  2. A plan in plain text: which physical quantities describe the question (motion, distance,
     rotation, speed, acceleration …), as up to 4 items, each tagged with one of six categories
  3. Tool calls; after each report: continue, revise the plan, or answer
on request            plain frames (get_frames), masks with visibility and size (get_segmentation_mask),
                      the full inventory (detect_all_entities)
```

The caption stays in the main agent's context through the answer (#36). Because the caption holds
no motion words (#34), it adds context without adding a motion prior.

**Why this shape.**

- **The steps** keep V5's staged decisions (one small decision per turn, which suits the 8B model),
  asked in physical terms as V4's first step was and as your P6 asks. There are no family or kind
  menus. The "what is it about" choice became a category tag on each plan item (#51), so it does a
  job: it selects the hint shown when that item is executed. Entities come before the plan (#37), so
  quantities are planned *for* named entities.
- **Step 3** is your P12: plan the high-level steps, then adapt to what the reports say.
- **The caption** comes from a subagent that knows nothing about the question, so it describes the
  video without steering towards an answer (#28).
- **The options** are visible from the start (#7).
- **The 8 keyframes** are shown automatically on the first turn, as in V5 (#32).
- **The inventory** runs only when the agent asks (#30). Other tools ground named entities lazily
  through the shared SAM3 cache.

## System prompt (#37, #38)

It opens with your framing:

> You are a 4D spatial reasoning agent with access to various measurement tools. You should plan to
> use these 3D tools effectively to measure the motion and interaction of entities in the 3D world,
> then answer based on your measurements from the 3D world.

The body mixes V4 and V5. From V4, how to read a measurement honestly (noise floors, spans,
unavailable is information, not proof of absence). From V5, its generic coverage of all three
benchmarks rather than V4's DSI-only wording. One added rule: when the caption says the clip is
first-person, the camera wearer is the observer; "the person" may be the camera, so use
`track_camera` (#38).

## Implementation notes

- **Captioner.** Same local Qwen3-VL-8B in a separate conversation. Its frame input is described
  below (#35). The caption is stored with the clip and reused by every question on that
  clip.
- **Entity step.** Names in words, singular, at most 5 words, with a distinguishing detail from the
  caption when several similar things are present ("the person in yellow"). The host validates
  only length and form. V5's word-overlap check against the question rejected 238 valid names on
  Dyn and is not carried over.
- **One decision per turn.** Each step ends in one small JSON object, in the style of V4's actions.

## Captioner input (#35)

The captioner gets the clip as a video, and Qwen3-VL's own sampler decides the frames. The installed
`Qwen3VLVideoProcessor` samples at 2 fps (minimum 4, maximum 768 frames), then fits the whole clip
into a total budget of about 25.2 M pixels (≈12k visual tokens) by shrinking every frame equally.

For a 10-second clip, 2 fps gives 20 frames at near-full resolution. Feeding all 300 raw frames
would squeeze each to about 380 × 220 px, and consecutive frames at 24–30 fps are near-duplicates.
The caption holds no motion words, so coverage and detail matter, not density. One call per clip,
cached; at most ≈1.8 GB of KV cache.

## Planning steps

```
Turn 1  entities   names in words; the camera is implicit                    (JSON, validated)
Turn 2  plan       one line "this question is about …" (#57), then
                   up to 4 numbered items in plain text, each with a category tag:
                     1. [counting] how many fish there are, and when each is visible
                     2. [entity motion] which way each fish swims, as the camera sees it
                   items are thinking, not tool calls                           (#49, #50, #51)
Turn 3+ act        a short thought + one tool call                           (JSON, validated)
        host       after each report: "Plan item 2 of 3 next: [entity motion] which way …
                   · calls used 4/14", plus the hint for that item's category
        agent      continue · "Revised plan:" + a new list · answer
```

**Decisions.**

- **The plan is plain text (#49).** It is chain of thought that grounds the next calls; the host does
  not validate it. Only the numbered lines are read, to show reminders. Every JSON plan or nested
  schema we tried failed (309 invalid envelopes in the rated pilot, 129 malformed calls in
  compositional DSI).
- **Plan items are thinking, not tool calls (#50).** They name quantities in words. The actual
  measurements are decided as the agent calls tools and reads reports. The host's reminders keep it
  on its plan; when something goes unexpectedly it rewrites the plan or pivots (#47).
- **Each item carries one of six category tags (#51):** camera motion · entity motion · entity and
  camera · entities and landmarks · counting · visual. The groups come from the census
  ([06](06_question_census.md)). The tag selects a short hint the host shows when that item is next.
  This replaces the separate "about" menu turn (#43).
- **Budget (#46):** at most 4 plan items and 14 tool calls, a warning on each of the last 3 calls, an
  early answer always allowed.
- **Visual items may pass the benchmark question and options to `ask_video` (#52).** Frames-only
  Qwen scores 79% on Dyn's visual questions, V5 71%.
- **Mixed questions** use `ask_video` for the visual fact and a measurement tool for the rest (#44).
- **Entities** are named in words, with `choose_segmentation_candidate` when SAM3 is ambiguous (#48,
  #55). Grounding retries are in [11](11_toolset.md) (#53).

**Tool visibility by phase (#79).** Hard limits by phase only. They are structural, so they cannot hide
a needed tool:

| Phase | What the agent can call |
|---|---|
| Turn 1, entities | nothing; it only names entities |
| Turn 2, plan | nothing, but it sees the tool list with one-line descriptions, so it plans measurable quantities |
| Act turns | all measurement tools, plus `get_frames` and `ask_video`; `choose_segmentation_candidate` only right after an ambiguity image (#55) |
| Answer turn | only the final answer |

There is no gating by plan item or category. The current item's category hint names the one or two
tools that usually fit, and the rest stay visible. V5's kind-scoped menus kept calls tidy but locked
the model out of the right tool after a wrong routing choice: 290 of 366 DSI subject questions were
measured in the scene frame and scored 40%, against 73% for the own-frame route. Gating by category
would also contradict #47. Full schemas are sent once; later turns list tool names with their
one-line descriptions.

**Category hints** (draft wording; these also carry the per-tool when / when-not of P11):

| Category | Hint shown with the item |
|---|---|
| camera motion | `track_camera` gives the camera's translation and turning in its start view. Zoom: `track_camera_zoom`. Returning view: `check_scene_revisit`. |
| entity motion | `track_entity` in the world frame, unless the question names the entity's own start (`subject`) or the camera's view (`camera`). Its turning is in the same report. Facing: `track_orientation`. |
| entity and camera | `track_entity(X, camera)`: distance, and how the camera sees X. `track_entity(X, subject)`: which side of X the camera is on. On-screen size and visibility: `get_segmentation_mask`. |
| entities and landmarks | `track_entity(X, others=[Y], subject)`: distance and side of Y relative to X over time. |
| counting | Find the instances (`get_segmentation_mask` or `detect_all_entities`); `track_multiple_entities` if a motion condition applies. You decide the count from the segmented frames. |
| visual | `ask_video`, to which you may pass the question and options; `get_frames` to look yourself. |

**Worked examples.**

| Question | Plan |
|---|---|
| "How many fish are swimming to the right?" | 1. [counting] how many fish, and when visible 2. [entity motion] each fish's direction as the camera sees it 3. [counting] check the right-movers in the segmented frames |
| "Relative to its starting orientation, how is the observer moving?" (DSI c3, orbit options) | 1. [camera motion] the camera's translation and turning |
| "Does the camera pan left before the skier jumps?" (Dyn) | 1. [visual] when the skier jumps 2. [camera motion] when the left pan starts 3. compare |
| "Does the dog move closer to the fence?" (Dyn landmark) | 1. [entities and landmarks] the dog's distance from the fence over time |
| "What direction is the bus moving?", with an option "there are no buses" (VLM4D false positive) | 1. [entity motion] whether a bus appears, and when 2. [entity motion] if so, its direction |
| "How is the people's location moving relative to his/her/its starting orientation?" (DSI c1) | 1. [entity motion] the person's motion and turning in its own starting facing |

