# Basketball 4D Tool-Use V1 Proposal

## Objective

Demonstrate that Qwen3-VL can answer natural-language metric motion questions by
using Open-D4RT as a 4D geometry expert on:

`demo/pstudio_mini/basketball_6/assets/input_video.mp4`

The first two target questions deliberately test different geometric quantities:

1. **Endpoint displacement**: "What is the metric distance between the starting
   and ending position of the basketball?"
2. **Distance travelled**: "How much distance did the basketball cover between
   the first and last frame?"

These terms must not be conflated. Endpoint displacement is one Euclidean norm;
distance travelled is the arc length of the trajectory and requires intermediate
positions.

## Feasibility with the current pipeline

The current pipeline already has most of the required pieces:

- `Qwen3VLBackend` presents sampled video frames and tool descriptions to Qwen.
- `ground_object(u, v, t)` maps a Qwen-grounded pixel to a D4RT `track_id`.
- `displacement(track_id, t0, t1)` computes endpoint separation.
- `trajectory_summary(track_id)` and `DemoGeometry.path_length(...)` compute
  accumulated visible trajectory length.
- `demo_data.json` contains predicted tracks and corresponding WorldTrack GT, so
  both questions can be scored without manual metric annotation.

The proof of concept is therefore feasible. The main missing work is not D4RT
inference; it is making semantic grounding and the measurement semantics robust.

## What the basketball bundle currently shows

The bundle has 64 frames at 15 FPS, resolution 640 x 360, and 31 sparse tracks.
The basketball is seeded near pixel `(346, 166)` in frame 0. Six duplicate entries
share this seed; track 5 is a representative one.

For predicted track 5:

- first 3D position: approximately `(0.161, -0.060, 2.289)` m
- last 3D position: approximately `(0.227, -0.194, 2.002)` m
- endpoint displacement: `0.3243 m`
- visible accumulated path length: `3.6049 m`
- predicted visible segments: frames `0-33` and `56-63`

For the corresponding GT track:

- endpoint displacement: `0.5625 m`
- visible accumulated path length: `3.1581 m`

This is a useful test case: the endpoint task is simple, while the path-length task
exercises temporal tracking, visibility handling, and accumulation. It also exposes
a real D4RT error rather than producing a contrived perfect demo.

## Proposed agent contract

Qwen should be responsible for semantics and planning. Deterministic tools should
be responsible for geometry and arithmetic.

### Qwen responsibilities

1. Understand that the referent is the orange basketball.
2. Understand whether the question asks for endpoint displacement or path length.
3. Choose a frame where the ball is clearly visible, preferably frame 0 for these
   questions.
4. Produce a point or box for the ball in that frame.
5. Select the appropriate measurement tool.
6. Report the tool result and its unit, including a visibility caveat when needed.

### Tool responsibilities

1. Resolve a semantic grounding to one or more nearby D4RT tracks.
2. Validate that the selected track stays spatially consistent with the ball.
3. Compute endpoint displacement or path length.
4. Handle visibility gaps explicitly.
5. Return confidence and diagnostics along with the metric answer.

Qwen should not manually subtract 3D coordinates or sum dozens of trajectory
segments. That would test language-model arithmetic instead of tool use.

## Recommended tool surface

### `ground_object`

Initial V1 input:

```json
{"label": "basketball", "u": 346, "v": 166, "t": 0}
```

Recommended output:

```json
{
  "object_handle": "basketball:0",
  "track_ids": [5, 6, 7, 8, 9, 10],
  "representative_track_id": 5,
  "pixel_distance": 1.7,
  "grounding_confidence": 0.94
}
```

Returning a small track ensemble is preferable to silently choosing among duplicate
surface-point tracks. The downstream result can use the median and expose the
ensemble spread as uncertainty.

### `endpoint_displacement`

```json
{
  "object_handle": "basketball:0",
  "t0": 0,
  "t1": 63
}
```

The tool internally obtains the two endpoint positions and computes:

`||position(t1) - position(t0)||_2`

This is logically equivalent to two D4RT position queries, but a high-level tool is
safer and easier to evaluate.

### `path_length`

```json
{
  "object_handle": "basketball:0",
  "t0": 0,
  "t1": 63,
  "visibility_policy": "visible_segments",
  "sampling_stride": 1
}
```

The tool sums consecutive 3D steps only when both frames are visible. It must return
the covered-frame fraction and gaps; otherwise "distance travelled over the whole
video" overclaims what was measured.

Recommended output:

```json
{
  "path_length_m": 3.6049,
  "visible_frames": 42,
  "total_frames": 64,
  "visible_segments": [[0, 33], [56, 63]],
  "coverage": 0.656,
  "measurement": "visible trajectory only"
}
```

### `inspect_track`

Return selected 2D projections at key frames, confidence, visible segments, and
ensemble consistency. This lets Qwen or the harness reject an obvious background
snap or identity switch before reporting a metric answer.

## Expected tool traces

### Endpoint displacement question

Minimal trace:

1. Qwen sees the ball in frame 0 and calls `ground_object`.
2. Tool returns a basketball object handle.
3. Qwen calls `endpoint_displacement(handle, 0, 63)`.
4. Tool returns approximately `0.324 m` from D4RT.
5. Qwen answers: "The basketball's start and end positions are approximately
   0.32 m apart."

The tool may internally make two position lookups, but Qwen does not need to perform
the arithmetic itself.

### Distance-travelled question

Minimal trace:

1. Qwen calls `ground_object` once.
2. Qwen calls `path_length(handle, 0, 63)`.
3. The tool evaluates all intermediate D4RT positions, splitting at visibility
   gaps.
4. Qwen answers: "Across the visible portions of the trajectory, the basketball
   travelled approximately 3.60 m; it was tracked in 42 of 64 frames."

If full-video distance is required, V1 should abstain or clearly state that the
occluded segment was not measured. Interpolating the missing segment should be a
separate, explicitly named policy and not the default.

## Required implementation changes

### Phase 1: deterministic basketball harness

1. Add `path_length` to the public tool schemas and dispatcher.
2. Make displacement and path-length results include visibility diagnostics.
3. Add a fixed `basketball_6` question set with the two prompts, GT answers, and
   tolerances.
4. Add a scripted grounding baseline at `(346, 166, frame 0)`.
5. Verify the exact calls and scores on CPU using the precomputed bundle.

This phase isolates the D4RT/tool ceiling from Qwen grounding and planning.

### Phase 2: Qwen native grounding

1. Present labelled frame samples, including exact frame indices 0 and 63.
2. Ask Qwen3-VL for a point or box on the basketball in frame 0.
3. Pass the box center plus several interior points to the grounding resolver.
4. Reject candidates beyond a configurable pixel distance.
5. Run Qwen with deterministic decoding and save the complete tool trace.

SAM is not required for the first attempt because the basketball is large, compact,
and visually distinctive. Add segmentation only if native Qwen grounding frequently
lands on the person, ball boundary, or background.

### Phase 3: robust object-level geometry

1. Replace the single nearest track with a local ensemble of ball surface tracks.
2. Aggregate 3D positions with a coordinate-wise median.
3. Report ensemble spread as geometric uncertainty.
4. Add track-validation diagnostics using reprojected 2D points at sampled frames.
5. Add optional temporal smoothing for path length, while keeping raw and smoothed
   values separately reported.

This phase matters more for path length than endpoint displacement because
frame-to-frame reconstruction noise always increases accumulated arc length.

### Phase 4: evaluation and ablations

Run at least these configurations:

1. Oracle grounding + GT geometry: harness sanity ceiling.
2. Oracle grounding + predicted D4RT geometry: geometry ceiling.
3. Qwen grounding + predicted D4RT geometry: complete V1.
4. Qwen-only answer without tools: semantic baseline.
5. Optional Qwen + SAM + D4RT: grounding ablation.

Score each stage independently:

- **Grounding:** point-in-ball / box IoU and chosen-track correctness.
- **Planning:** correct tool selection for displacement versus path length.
- **Geometry:** absolute and relative metric error against GT.
- **Answering:** numeric extraction, correct unit, and visibility qualification.
- **Reliability:** success rate over multiple prompt paraphrases.

## Success criteria for V1

The V1 demo is successful if:

1. Qwen selects `endpoint_displacement` for endpoint wording and `path_length` for
   distance-travelled wording in at least 90% of deterministic prompt variants.
2. Qwen grounds the basketball to the correct local track ensemble in at least 90%
   of trials.
3. The final numeric answer exactly reflects the tool output and uses meters.
4. Path-length answers disclose incomplete visibility.
5. The saved trace makes semantic, grounding, and geometry failures separable.

The predicted D4RT numbers do not need to match GT perfectly for this first demo;
the goal is to prove correct orchestration and quantify where error enters.

## Risks and mitigations

### Sparse grounding

The demo bundle contains only 31 sparse tracks. A good Qwen point can still snap to
the wrong object if no track lies sufficiently close. Use a distance threshold,
return multiple candidates, and abstain when grounding is unsupported.

### Duplicate ball tracks

Several entries share the ball seed. Treat them as an ensemble rather than distinct
objects, or deduplicate them during bundle creation.

### Occlusion and visibility gaps

The predicted ball track is not visible for frames 34-55. Report visible-only path
length by default. Do not silently bridge the gap.

### Path-length inflation

Accumulated distance is sensitive to per-frame 3D jitter. Report raw path length,
optionally add a smoothed estimate, and evaluate both against GT. Never apply
smoothing without naming the policy.

### Metric scale leakage

This demo's predicted trajectories are globally aligned to WorldTrack GT scale.
That is acceptable for tool-use prototyping, but it is not a fully in-the-wild
monocular metric claim. The evaluation report must state this explicitly.

### Tool-call parsing

The current backend embeds JSON schemas in text and parses XML-like tool-call tags.
Validate this against the installed Qwen3-VL/Transformers version. Prefer the
processor's native chat-template tool interface if supported.

## Scope estimate

- Phase 1: small, approximately one focused implementation day.
- Phase 2: small-to-moderate, approximately one to three days including GPU and
  prompt debugging.
- Phase 3: moderate, approximately three to five days.
- Phase 4: moderate, approximately two to four days for clean experiments and
  analysis.

A persuasive single-scene prototype is therefore roughly one week of focused work.
A benchmark-quality system with multiple objects, scenes, grounding annotations,
and robust uncertainty handling is several additional weeks.

## Recommended immediate next step

Implement Phase 1 first and freeze its two deterministic traces. It will establish
that the desired questions are answerable with the present bundle and will prevent
Qwen prompt debugging from being confused with D4RT geometry or path-definition
bugs. Then run Phase 2 without SAM and add SAM only if measured grounding failures
justify it.
