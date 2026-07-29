# Phase 2 — add an isolated Qwen bbox/points grounding tool and evaluate it on the fixed 25-question DSI subset

## 1. Goal and hypothesis

Give the reasoning agent a new visual grounding action with two explicit modes:

- `bbox`: return one tight box for a requested object.
- `points`: return a requested number of 2D points for object parts or static scene landmarks.

The grounding inference must use a fresh, one-frame, grounding-only Qwen conversation. It must not contain the DSI question, answer choices, agent reasoning, D4RT measurements, or prior tool history.

The reasoning agent will see the returned coordinates and descriptions, but it will pass an immutable `grounding_id` to D4RT instead of manually copying coordinates.

The hypothesis is:

> Qwen grounds many ordinary objects well when localization is isolated from long-form 4D reasoning. Separating the two contexts should reduce on-the-fly coordinate errors without taking semantic/tool-planning responsibility away from the main agent.

This is an experiment, not a guaranteed accuracy improvement. A technically correct negative result must be reported honestly.

## 2. Evidence motivating the design

The isolated grounding probe over the existing 25-question traces found:

- 56 unique `(question, t_src, label)` requests;
- 50 boxes and 6 explicit nulls;
- no parse/runtime failures;
- very different boxes from the agent-loop boxes;
- visibly better boxes for several obvious whole objects;
- continued weakness on fine-grained front/back parts and ambiguous background labels.

The failed prompt-only forced-grounding experiment found:

- forced-grounding D4RT agent: 7/25;
- previous committed D4RT agent: 10/25;
- unchanged Qwen-only baseline: 13/25;
- four unanswered questions;
- 37.5% of model responses containing multiple actions;
- 176.6 minutes total, with repeated strict and relaxed loops.

The conclusion is not “Qwen grounding is solved.” The conclusion is that grounding quality is context-sensitive and should be isolated behind a tool boundary.

## 3. Prerequisites and stop/go gate

Do not start Phase 2 until Phase 1 is complete.

Required state:

- current branch is `feat/qwen-grounding-tool-orchestration`;
- worktree is clean;
- all Phase 1 CPU tests pass;
- the three-question Phase 1 Blackwell pathology replay passes every structural gate;
- the replay has no strict-attempt exhaustion, phantom evidence, or multi-action effective turns;
- the Phase 1 scratch artifacts and job log are recorded;
- no canonical DSI result has been overwritten.

Verify:

```bash
git branch --show-current
git status --short --branch
git log --oneline --decorate -8
```

Run the complete relevant test suite again:

```bash
/cluster/work/igp_psr/spanwar/envs/d4rt_bw/bin/python -m unittest \
  d4rt_agent.test_simple_v2 \
  d4rt_agent.test_dsi_bench \
  d4rt_agent.test_qwen_grounding_debug \
  d4rt_agent.test_orchestration_audit
```

Stop if Phase 1 is not proven stable. Do not mask an orchestration failure with grounding-tool prompt changes.

## 4. Scope

### In scope

- One new model-facing `ground_with_qwen` action.
- `bbox` and `points` modes.
- A fresh grounding-only context for every uncached request.
- Reuse of the already loaded Qwen checkpoint in the same process.
- Strict parsing, bounds checks, and explicit `not_found`.
- Host-assigned immutable `qg_N` and point IDs.
- Cache reuse for identical grounding requests.
- D4RT queries by `grounding_id`.
- Existing five-point bbox policy fixed to host `ensemble5`.
- Exact D4RT trajectories for explicit point requests.
- Grounding provenance in evidence, traces, reports, and overlays.
- Independent bbox/points smoke testing.
- Full evaluation on the exact same 25-question manifest.
- Blackwell RTX PRO 6000 jobs only.

### Explicitly deferred

Do not add these policies in this phase:

- task-specific grounding requests inserted by the host;
- automatic duplicate D4RT-query rejection;
- a new per-question grounding-call budget beyond the existing global step budget;
- DSI-specific recovery after `not_found`;
- new partial-visibility heuristics;
- removal/redesign of the strict/relaxed retry;
- forced final answers;
- token-level generation stopping;
- textual validation of the final explanation;
- task-specific camera-motion, facing, or answer-selection rules;
- any A100 run.

The main agent must remain responsible for deciding what to ground and why.

## 5. Architecture

### 5.1 Responsibilities

The main reasoning agent owns:

- understanding the question and options;
- selecting the object, part, or scene landmark;
- selecting `bbox` versus `points`;
- selecting `t_src`, `t_tgt`, and `t_cam`;
- deciding how many points are useful;
- interpreting D4RT trajectories;
- choosing the final answer.

The grounding tool owns only:

- viewing one exact sampled source frame;
- resolving a short natural-language request in that frame;
- returning a tight box, requested points, or `not_found`;
- never reasoning about the DSI answer.

The host owns:

- schema validation;
- normalized coordinate validation;
- grounding IDs and point IDs;
- cache keys;
- clip/source-frame binding;
- resolving a grounding ID for D4RT;
- all evidence and provenance.

### 5.2 One physical checkpoint, two logical contexts

Do not load a second 8B model copy unless memory measurements later prove it necessary and the user approves it.

`dsi_bench_run.py` should continue constructing one `OfflineQwen`. Pass that same object into the grounding tool. Each grounder invocation calls it sequentially with a brand-new message list.

Required isolation:

```text
main-agent generation:
  shared tool protocol + DSI system prompt + 32 frames + question + real history

grounder generation:
  grounding-only system prompt + exactly one selected frame + short request
```

The Python object may be shared; the message context must not be shared.

Extend `OfflineQwen.generate` with an optional per-call `max_new_tokens` override so grounding calls can use a small budget without changing the main agent’s 768-token budget:

```python
def generate(
    self,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int | None = None,
) -> str:
    ...
```

Existing callers must behave identically when the override is omitted.

## 6. Files to add or modify

Expected implementation surface:

| File | Change |
| --- | --- |
| `d4rt_agent/qwen_grounding_tool.py` | New grounding prompt, parsing, validation, cache-key helpers, and tool class |
| `d4rt_agent/prompts/qwen_grounding_tool.md` | New one-frame grounding-only system prompt |
| `d4rt_agent/simple_v2.py` | Four-action orchestration, grounding registry/cache integration, query resolution |
| `d4rt_agent/simple_v2_contracts.py` | New action schema and new `query_d4rt` schema |
| `d4rt_agent/simple_v2_backend.py` | Shared explicit-UV query path; bbox aggregation and separate point tracks |
| `d4rt_agent/prompts/simple_v2_system.md` | Teach ground → D4RT → math → final |
| `d4rt_agent/prompts/dsi_bench_system.md` | Replace raw-coordinate instructions/examples with grounding IDs |
| `d4rt_agent/dsi_bench_run.py` | Record grounding configuration/run metadata; reuse one Qwen |
| `d4rt_agent/dsi_bench_report.py` | New qg trace rendering and exact-frame bbox/point overlays |
| `d4rt_agent/test_simple_v2.py` | Contracts, isolation, caching, orchestration, backend output tests |
| `d4rt_agent/test_dsi_bench.py` | DSI prompt/final-answer integration tests |
| `d4rt_agent/test_qwen_grounding_tool.py` | Focused strict parser and grounder tests |
| `d4rt_agent/test_dsi_bench_compare.py` | Synthetic comparison/counting tests |
| `d4rt_agent/qwen_grounding_tool_debug.py` | Required no-D4RT smoke runner |
| `d4rt_agent/fixtures/qwen_grounding_tool_smoke.json` | Exact six smoke requests; prevents post-result fixture selection |
| `d4rt_agent/dsi_bench_compare.py` | CPU-only comparison and final findings metrics |
| `scripts/run_qwen_grounding_tool_debug_blackwell.slurm` | Required Blackwell-only smoke launcher |
| `scripts/run_dsi_bench_blackwell.slurm` | Reuse Phase 1 overrides; print grounding configuration |

Keep `qwen_grounding_debug.py` intact as the historical trace-replay probe. Reuse its proven parsing, normalization, and drawing ideas, but do not turn that historical script into the live tool.

## 7. Model-facing action contracts

Phase 2 exposes exactly four actions:

1. `ground_with_qwen`
2. `query_d4rt`
3. `python_math`
4. `final_answer`

The main agent must no longer generate raw D4RT coordinates.

### 7.1 `ground_with_qwen`

Schema:

```json
{
  "name": "ground_with_qwen",
  "description": "Ground an object box or requested 2D points in one sampled source frame using an isolated one-frame Qwen context.",
  "parameters": {
    "type": "object",
    "additionalProperties": false,
    "required": ["mode", "t_src", "request", "justification"],
    "properties": {
      "mode": {"type": "string", "enum": ["bbox", "points"]},
      "t_src": {"type": "integer", "minimum": 0, "maximum": 31},
      "request": {"type": "string"},
      "count": {"type": "integer", "minimum": 1, "maximum": 8},
      "justification": {"type": "string"}
    }
  }
}
```

Validation rules:

- only `mode`, `t_src`, `request`, `count`, and `justification` are allowed;
- `request` must be non-empty after trimming.
- Maximum request length: 500 characters.
- `justification` follows the existing non-final action requirement.
- `count` is required for `mode="points"`.
- `count` must be absent for `mode="bbox"`.
- Point count is 1–8 inclusive.
- `t_src` is a sampled-frame index in 0–31.
- Reject any raw coordinate field supplied by the agent.
- Reject unknown mode values.

Valid examples:

```json
{"action":"ground_with_qwen","arguments":{"mode":"bbox","t_src":0,"request":"the red car closest to the camera","justification":"Track the nearest red car over the clip."}}
```

```json
{"action":"ground_with_qwen","arguments":{"mode":"points","t_src":0,"request":"runner's chest and back","count":2,"justification":"Measure two parts that define the runner's facing direction."}}
```

```json
{"action":"ground_with_qwen","arguments":{"mode":"points","t_src":0,"request":"five static points on the distant background","count":5,"justification":"Use stationary landmarks to infer camera motion."}}
```

```json
{"action":"ground_with_qwen","arguments":{"mode":"points","t_src":8,"request":"three distinct corners on the building","count":3,"justification":"Track rigid, separated background landmarks."}}
```

### 7.2 Phase 2 `query_d4rt`

Replace the model-facing raw-box schema. The new schema is:

```json
{
  "name": "query_d4rt",
  "description": "Query D4RT using an immutable grounding previously returned by ground_with_qwen.",
  "parameters": {
    "type": "object",
    "additionalProperties": false,
    "required": ["grounding_id", "t_tgt", "t_cam", "justification"],
    "properties": {
      "grounding_id": {"type": "string"},
      "point_ids": {
        "type": "array",
        "items": {"type": "string"}
      },
      "t_tgt": {
        "type": "array",
        "items": {"type": "integer", "minimum": 0, "maximum": 31}
      },
      "t_cam": {"type": "integer", "minimum": 0, "maximum": 31},
      "justification": {"type": "string"}
    }
  }
}
```

The agent-facing schema must contain none of:

- `label`;
- `bbox_2d_1000`;
- `points_2d_1000`;
- `t_src`;
- `point_mode`.

Host validation rules:

- only `grounding_id`, `point_ids`, `t_tgt`, `t_cam`, and `justification` are allowed;
- `grounding_id` must exist in authoritative evidence.
- It must identify a `ground_with_qwen` result.
- The grounding must belong to the current clip and sampled-frame mapping.
- The grounding status must be `ok`, not `not_found`.
- `t_src` comes only from the stored grounding.
- `point_ids` is invalid for bbox groundings.
- For a points grounding, omitted `point_ids` means all returned points.
- A supplied point list must be non-empty, unique, and a subset of that grounding’s point IDs.
- `t_tgt` must be non-empty, unique, and inside 0–31.
- `t_cam` must be inside 0–31.
- The main agent cannot override or mutate stored geometry.

Valid bbox use:

```json
{"action":"query_d4rt","arguments":{"grounding_id":"qg_1","t_tgt":[0,31],"t_cam":0,"justification":"Measure the grounded runner's endpoint motion in one fixed camera frame."}}
```

Valid selected-point use:

```json
{"action":"query_d4rt","arguments":{"grounding_id":"qg_2","point_ids":["p1","p2"],"t_tgt":[0,31],"t_cam":0,"justification":"Track the grounded chest and back points separately."}}
```

## 8. Grounding-only prompt and raw output

Create `d4rt_agent/prompts/qwen_grounding_tool.md`.

The grounder sees:

- one source-frame image;
- mode;
- request;
- count for points;
- coordinate/output rules.

It does not see:

- the DSI question;
- answer options;
- any other frame;
- prior agent reasoning;
- the main tool schemas;
- evidence IDs;
- D4RT output;
- final-answer instructions.

### 8.1 Bbox prompt

Use the successful isolated-probe convention:

```text
Ground the target named "<request>" in this image inside one tight bounding box.

Return exactly one JSON object and no other text:
{"bbox_2d_1000":[x_min,y_min,x_max,y_max]}

Normalize coordinates to integers from 0 to 1000 with origin at the top-left.
If the requested target is not visible, return:
{"bbox_2d_1000":null}
```

### 8.2 Points prompt

Required behavior:

- return up to the requested count;
- return fewer rather than invent unreliable points;
- use points securely inside visible surfaces;
- avoid uncertain silhouettes and boundaries;
- for named parts, return separately described points in requested order where possible;
- for multiple points on one object, make them spatially distinct;
- for background requests, use rigid high-contrast landmarks spread across the frame;
- prefer building/window corners, poles, rocks, trunks, and fixed edges;
- avoid people, animals, moving vehicles, reflections, shadows, sky, and moving foliage.

Exact output form:

```json
{
  "points_2d_1000": [
    {"xy":[312,284],"description":"left corner of the upper window"},
    {"xy":[571,301],"description":"right corner of the roof"}
  ]
}
```

If none can be located:

```json
{"points_2d_1000":null}
```

The grounder must not emit `grounding_id` or `point_id`; the host owns both.

Use a small deterministic generation budget, initially 192 tokens for points and 96 for bbox. Keep greedy decoding and seed 42.

## 9. Strict parsing and validation

Implement parsing in `qwen_grounding_tool.py` with `json.JSONDecoder.raw_decode`, following the proven `parse_detection` pattern.

### 9.1 Bbox validation

Accept exactly one `bbox_2d_1000` field whose value is:

- `null`; or
- a four-number list `[x_min, y_min, x_max, y_max]`.

For a box require:

- no booleans;
- all finite;
- all in `[0,1000]`;
- `x_max > x_min`;
- `y_max > y_min`.

Store normalized values as floats or integers consistently. Do not silently clamp invalid model output.

### 9.2 Points validation

Accept exactly one `points_2d_1000` field whose value is:

- `null`; or
- a list of point objects.

For every point require:

- exactly `xy` and `description`;
- `xy` has two finite numeric values in `[0,1000]`;
- non-empty string description;
- description length at most 200 characters.

Additional rules:

- zero returned points becomes `not_found`;
- 1 through requested `count` points is valid;
- fewer than requested is valid and explicitly reported;
- more than requested is invalid;
- duplicate coordinates are invalid;
- the host assigns `p1`, `p2`, … in returned order.

### 9.3 Parse failure

A malformed grounder response is a tool execution failure:

- save the raw grounder response in the trace;
- create no `qg_N`;
- do not advance the qg counter;
- tell the reasoning agent that no grounding ID was created;
- list existing qg/evidence IDs in the ledger;
- suggest rephrasing the target or selecting another `t_src` without giving DSI-specific advice.

Do not fall back to extracting coordinates from prose.

## 10. Grounding evidence shape

The first uncached valid grounding call creates an immutable host record.

### 10.1 Bbox result

Model-facing view:

```json
{
  "grounding_id": "qg_1",
  "status": "ok",
  "mode": "bbox",
  "t_src": 0,
  "request": "the red car closest to the camera",
  "bbox_2d_1000": [447,350,615,580]
}
```

Not found:

```json
{
  "grounding_id": "qg_2",
  "status": "not_found",
  "mode": "bbox",
  "t_src": 0,
  "request": "the requested object",
  "bbox_2d_1000": null
}
```

### 10.2 Points result

```json
{
  "grounding_id": "qg_3",
  "status": "ok",
  "mode": "points",
  "t_src": 0,
  "request": "runner's chest and back",
  "requested_count": 2,
  "returned_count": 2,
  "points_2d_1000": [
    {"point_id":"p1","xy":[477,374],"description":"center of runner's chest"},
    {"point_id":"p2","xy":[451,382],"description":"visible rear torso surface"}
  ]
}
```

### 10.3 Host-only provenance

The stored evidence/trace should additionally include:

- resolved video path;
- sampled-to-original mapping or a stable digest of it;
- original source-frame index;
- normalized request;
- cache-key digest;
- raw grounder response;
- grounding prompt version;
- Qwen model path;
- generation seed and token budget;
- inference wall time;
- cache-hit status.

Do not expose the clip key, cache key, prompt text, model path, or raw grounder response to the main reasoning agent. Keep them in stored evidence/trace for diagnostics. The model-facing result contains the parsed coordinates, descriptions, source frame, request, mode/count, status, and qg/point IDs.

A valid `not_found` response is real evidence and receives a qg ID. D4RT must reject using it.

## 11. Cache identical grounding requests

Use a canonical key:

```text
(clip identity, sampled mapping, t_src, mode, normalized request, count-or-null)
```

Normalize requests by:

1. Unicode NFKC normalization.
2. Trim leading/trailing whitespace.
3. Collapse internal whitespace to one space.
4. Case-fold.

Do not perform semantic synonym rewriting.

Cache requirements:

- the cache is owned by one `SimpleV2Orchestrator` instance and is reset when the next benchmark question creates a new orchestrator;
- geometry is immutable after registration;
- an identical request does not invoke Qwen again;
- an identical request returns the same `qg_N`;
- qg counter does not advance on a cache hit;
- trace marks `cache_hit=true` and `reused_evidence_id=qg_N`;
- ledger says no new evidence was created and names the reused ID;
- `not_found` results are cached too;
- a different frame, mode, request, or count produces a different key.

Phase 1’s execution-result interface may need a small refactor so an accepted cached call can reuse an existing evidence ID without creating a duplicate entry. Make this explicit with a structured execution outcome rather than special-casing strings in `solve`.

Suggested shape:

```python
@dataclass
class ActionExecution:
    result: dict[str, Any]
    response_content: list[dict[str, Any]] | None
    evidence_prefix: str | None
    reused_evidence_id: str | None = None
```

For a normal new grounding, `evidence_prefix="qg"` and `reused_evidence_id=None`.
For a cache hit, `evidence_prefix=None` and `reused_evidence_id="qg_1"`.

## 12. Bind each grounding to its clip and source frame

Create a stable clip key from:

- resolved video path;
- total decoded frames;
- width and height;
- exact `sampled_to_original` mapping.

Store the key in host-only grounding provenance.

Before D4RT execution, compare the grounding’s clip key with the current `SampledVideo`. Reject on mismatch.

The main agent never supplies `t_src` to D4RT in Phase 2. The host resolves it from the grounding. This prevents:

- applying a frame-0 box as if grounded in frame 15;
- carrying a qg ID into another question/clip;
- changing the geometry after seeing a D4RT result.

## 13. Refactor the D4RT backend around explicit UV queries

The current `LiveD4RTBackend.query` accepts a box, creates centroid/ensemble points, runs D4RT, and aggregates those points into one prediction per target.

Refactor without duplicating model invocation code.

### 13.1 Shared low-level path

Create a private helper that accepts:

- explicit source points with pixel and normalized UV;
- one `t_src`;
- ordered `t_tgt`;
- one `t_cam`.

It should:

1. construct the flattened D4RT query tensors;
2. invoke `_run_model_for_queries` once;
3. reshape outputs as `[num_points, num_targets, ...]`;
4. return per-point/per-target raw predictions.

Keep the existing order consistent:

- points are the outer dimension;
- target frames are the inner dimension;
- reshape and `np.tile`/`np.repeat` must match.

Add a fake-output unit test that proves point/target rows are not transposed.

### 13.2 Bbox grounding

For a bbox qg:

- use host-configured `ensemble5`;
- derive the existing centroid plus four seeded offsets inside the box;
- require the existing 3-of-5 visible policy;
- preserve current top-level output fields used by `python_math`:
  - `predictions`;
  - `math_trajectory_aligned_xyz_m`;
  - `math_visibility`;
  - `visibility_coverage`;
- add `grounding_id`, `grounding_mode="bbox"`, request, and source provenance;
- keep internal ensemble coordinates hidden from Qwen through `_redact_host_policy`.

The main agent must never select `ensemble5`; it remains host policy.

### 13.3 Explicit points grounding

For a points qg:

- convert each selected `[x,y]` in 0–1000 coordinates to exact pixel and D4RT normalized UV;
- query exactly those points;
- do not add centroid offsets;
- do not average distinct points together;
- preserve one independent trajectory per `point_id`.

Return:

```json
{
  "grounding_id": "qg_3",
  "grounding_mode": "points",
  "point_ids": ["p1","p2"],
  "t_src": 0,
  "t_tgt": [0,31],
  "t_cam": 0,
  "point_tracks": [
    {
      "point_id": "p1",
      "description": "center of runner's chest",
      "source_xy_1000": [477,374],
      "source_pixel_uv": [305.1,179.2],
      "predictions": [
        {
          "sampled_frame_index": 0,
          "benchmark_aligned_xyz_m": [0.1,0.2,3.4],
          "visible": true,
          "target_uv_norm": [0.47,0.37]
        }
      ],
      "math_trajectory_aligned_xyz_m": [[0.1,0.2,3.4],[0.0,0.0,0.0]],
      "math_visibility": [true,false],
      "visibility_coverage": 0.5
    }
  ]
}
```

Each `predictions` row should retain the same useful raw/aligned xyz, visibility probability, confidence, and target UV fields as the current individual point result.

For restricted math:

- invalid positions remain `[0,0,0]`;
- the matching visibility entry must be false;
- prompts must tell the model to bind the visibility mask with the trajectory;
- paths through `point_tracks` must work with existing binding resolution.

### 13.4 Compatibility

Keep an internal legacy bbox entry point if existing non-agent code/tests require it, but do not expose raw boxes in the Phase 2 action schema.

Run:

```bash
rg -n "\.query\(|query_d4rt|bbox_2d_1000" d4rt_agent scripts
```

Update every real caller and test deliberately. Do not leave a silent call site using the old signature.

## 14. Integrate the tool into single-action orchestration

The expected live sequence is:

```text
ASSISTANT TURN 1
I need a reliable box around the runner.
{"action":"ground_with_qwen",...}

HOST TURN
Tool result qg_1: parsed coordinates
HOST EVIDENCE STATE: qg_1 exists

ASSISTANT TURN 2
The runner is grounded; now I will query its endpoints.
{"action":"query_d4rt","arguments":{"grounding_id":"qg_1",...}}

HOST TURN
Tool result d4rt_1: real trajectory
HOST EVIDENCE STATE: qg_1 and d4rt_1 exist

ASSISTANT TURN 3
{"action":"python_math",...}

ASSISTANT TURN 4
{"action":"final_answer",...}
```

Requirements:

- grounding inference occurs only after the host accepts `ground_with_qwen`;
- the main agent does not see the raw grounder conversation;
- grounder output cannot contain a second main-agent action;
- Phase 1 first-action boundaries apply unchanged;
- qg IDs appear in the authoritative ledger;
- D4RT results name the qg/point IDs that generated them;
- a final answer may cite qg evidence in addition to the required D4RT evidence;
- DSI’s existing requirement for at least one `d4rt_*` remains unchanged.

## 15. Prompt changes

Update both tool-using prompts.

### 15.1 Remove raw-coordinate instructions

Remove instructions telling the reasoning agent to:

- visually write `bbox_2d_1000`;
- copy or reuse raw coordinates;
- put `label`, `bbox_2d_1000`, or `t_src` into `query_d4rt`.

Retain explanations of:

- selecting a good source frame;
- object motion versus camera motion;
- fixed versus varying `t_cam`;
- scalar versus vector comparisons;
- restricted math;
- final-answer evidence.

### 15.2 Teach grounding choice

Teach:

- use bbox for a whole visible object;
- use points for named parts or static landmarks;
- ask for a semantically precise target;
- prefer distinct rigid points for background motion;
- use returned `grounding_id` rather than copying numbers;
- inspect status and returned count;
- if `not_found`, rephrase or choose a frame where the target is visible;
- points from one grounding are separately tracked.

### 15.3 Complete separated-turn examples

At least one complete example must show:

```text
ground_with_qwen bbox
→ real qg host result
→ query_d4rt by ID
→ real d4rt result
→ python_math
→ final_answer
```

At least one example must show:

```text
ground_with_qwen points for static background
→ query selected/all point IDs
→ reason over separate point tracks
```

At least one example must show fewer points than requested or `not_found` followed by a new single action. Do not show several assistant actions in one example turn.

## 16. Tests

### 16.1 Action contract tests

Assert:

- exactly four model-facing tools;
- `ground_with_qwen` validates both modes;
- point count is conditionally required and bounded 1–8;
- bbox mode rejects count;
- query schema requires grounding ID;
- query schema has no raw geometry, label, `t_src`, or point mode;
- unknown/foreign/not-found qg IDs are rejected;
- bbox qg rejects point IDs;
- point qg accepts all points or a valid subset;
- duplicate/unknown point IDs are rejected.

### 16.2 Parser tests

Test bbox:

- valid integer box;
- valid float box;
- null;
- reversed/zero-area box;
- out-of-bounds coordinate;
- NaN/Infinity;
- boolean;
- extra keys;
- JSON in a code fence;
- prose before strict JSON;
- no valid JSON.

Test points:

- valid exact count;
- valid fewer-than-requested count;
- null/empty;
- too many;
- duplicate coordinates;
- missing/empty description;
- wrong coordinate length;
- out of bounds/non-finite;
- extra keys;
- malformed JSON.

### 16.3 Isolation tests

Use a spy/fake Qwen. Prove an uncached grounding call receives:

- exactly one image;
- exactly the selected sampled frame;
- grounding-only prompt text;
- request, mode, and count;
- no DSI question;
- no option text;
- no prior assistant reasoning;
- no D4RT result;
- no main action schemas.

Prove the same `OfflineQwen` object is reused and no second checkpoint load is triggered.

### 16.4 Cache tests

Assert:

- identical canonical requests call Qwen once and return the same qg ID;
- whitespace/case normalization hits the same key;
- different frame/mode/count/request misses;
- cached `not_found` reuses the same qg ID;
- qg counter has no gap;
- trace and ledger distinguish new versus reused evidence;
- cached evidence remains immutable.

### 16.5 Backend tests

With fake D4RT outputs, verify:

- explicit point coordinates map correctly from 0–1000 to pixel and normalized UV;
- point/target reshape order is correct;
- bbox still uses five host points and 3-of-5 aggregation;
- point mode uses no hidden offsets;
- two points remain two tracks;
- a point’s invisible target is zeroed only in math trajectory and masked false;
- provenance fields identify qg and p IDs;
- bbox backward-compatible result paths still work;
- restricted math can bind point tracks and visibility arrays.

### 16.6 Orchestration tests

Script:

1. new grounding;
2. D4RT using qg;
3. math;
4. final.

Assert single-action history and evidence order:

```text
qg_1 → d4rt_1 → math_1 → final
```

Also script:

- not_found then a rephrased grounding;
- malformed grounder response then recovery;
- cache hit then D4RT;
- attempted coordinate mutation;
- attempted cross-clip qg use;
- unknown point ID.

Every rejection must create no evidence and return the real ledger.

### 16.7 Report tests

Generate temporary images/results and verify:

- bbox is drawn on exact `t_src`;
- each point is drawn and labelled with its point ID;
- combined same-frame detections are not duplicated across frames;
- qg provenance appears in trace/report;
- cached requests do not create duplicate geometry;
- a not-found grounding is represented in text without drawing fake geometry;
- old result records remain readable.

### 16.8 CPU validation commands

Run focused new tests first:

```bash
/cluster/work/igp_psr/spanwar/envs/d4rt_bw/bin/python -m unittest \
  d4rt_agent.test_qwen_grounding_tool \
  d4rt_agent.test_dsi_bench_compare
```

Then run the full relevant suite:

```bash
/cluster/work/igp_psr/spanwar/envs/d4rt_bw/bin/python -m unittest \
  d4rt_agent.test_simple_v2 \
  d4rt_agent.test_dsi_bench \
  d4rt_agent.test_qwen_grounding_debug \
  d4rt_agent.test_orchestration_audit \
  d4rt_agent.test_qwen_grounding_tool \
  d4rt_agent.test_dsi_bench_compare
```

Compile all modules and check both edited launchers:

```bash
/cluster/work/igp_psr/spanwar/envs/d4rt_bw/bin/python -m compileall -q d4rt_agent
bash -n scripts/run_dsi_bench_blackwell.slurm
bash -n scripts/run_qwen_grounding_tool_debug_blackwell.slurm
```

Do not submit the smoke or full GPU run until every command exits 0.

## 17. Grounding overlays and report output

The final results must include reviewable images for every frame where grounding was requested.

Extend report generation to create:

```text
results/dsi_bench/grounding_frames/<question_id>_f<sampled-frame>.jpg
```

Each image should:

- use the exact sampled source frame;
- combine all new Qwen groundings requested on that frame;
- draw bbox groundings as rectangles;
- draw point groundings as labeled circles;
- include `qg_N`, request text, status, and point IDs in a readable legend;
- use consistent colors per qg ID;
- draw only live Phase 2 qg geometry, not old agent-generated boxes.

Retain or update the per-question grounding contact sheet under `groundings/`.

Add a report flag:

```text
--refresh-groundings
```

It must rebuild grounding frames/sheets even when old JPEGs exist. It should not need to rebuild videos, GIFs, or ordinary contact sheets.

The Markdown report should expose:

- grounder request;
- mode/count;
- qg ID and cache status;
- parsed coordinates/descriptions;
- exact source frame;
- D4RT qg/point provenance;
- raw grounder response inside a collapsed diagnostic section;
- raw versus effective main-agent response diagnostics from Phase 1.

## 18. Independent Blackwell grounding smoke test

Before loading D4RT for the full run, exercise the live tool alone.

### 18.1 Scratch path

Use:

```text
/cluster/scratch/spanwar/tmp/d4rt_agent/phase_2_grounding_tool_smoke
```

Use a fresh child directory tied to the current Git commit.

### 18.2 Required request fixtures

Use these exact six fixtures from the unchanged 25-question manifest:

| Question ID | `t_src` | Mode | Request | Count |
| --- | ---: | --- | --- | ---: |
| `CameraBench_c0_camerabench_m0jmssq5ptw.3.12_ccd85243` | 0 | `bbox` | `runner` | — |
| `llava178k_c5_videos_youtube_video_2024_ytb_16mzjk4aojs_video-scene-00000_b36d2eb1` | 0 | `bbox` | `harvester` | — |
| `CameraBench_c0_camerabench_m0jmssq5ptw.3.12_ccd85243` | 0 | `points` | `runner's chest and back` | 2 |
| `CameraBench_c2_camerabench_u35wis62r2m.1.4_8c093924` | 0 | `points` | `five static points on the distant rigid background structure` | 5 |
| `CameraBench_c2_camerabench_u35wis62r2m.1.4_8c093924` | 0 | `points` | `three distinct corners on the central building structure` | 3 |
| `CameraBench_c0_camerabench_m0jmssq5ptw.3.12_ccd85243` | 0 | `bbox` | `the red car closest to the camera` | — |

The final fixture is intentionally absent from the runner frame and exercises possible `not_found`. Do not replace these fixtures after seeing the answers; doing so would turn the smoke check into prompt tuning.

Store the table as a tracked JSON request fixture at:

```text
d4rt_agent/fixtures/qwen_grounding_tool_smoke.json
```

The smoke runner must accept:

```text
--manifest PATH
--requests PATH
--results-dir PATH
--qwen-model PATH
--seed 42
--force
```

It resolves each question ID through the supplied manifest, samples the clip with `sample_video_cpu`, invokes the same `QwenGroundingTool` used by the live agent, writes one atomic JSON per request, and creates the overlays. It must not load D4RT.

Each smoke record must save:

- source frame;
- request and mode/count;
- raw grounder reply;
- parsed result/status;
- wall time;
- overlay.

### 18.3 GPU requirement

Submit only through a launcher requesting:

```text
#SBATCH --partition=cuda13pr.4h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --constraint=EPYC_9654
```

Confirm with `scontrol show job` and the log’s `nvidia-smi` output. Do not run the smoke test on A100.

Submit with a fresh scratch directory:

```bash
PHASE2_SMOKE_DIR="/cluster/scratch/spanwar/tmp/d4rt_agent/phase_2_grounding_tool_smoke/run_$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=12 HEAD)"
test ! -e "$PHASE2_SMOKE_DIR"
mkdir -p "$PHASE2_SMOKE_DIR"
sbatch --parsable \
  --export=ALL,RESULTS_DIR="$PHASE2_SMOKE_DIR",MANIFEST_PATH=/cluster/work/igp_psr/spanwar/Open-d4rt/d4rt_agent/results/dsi_bench/manifest.json,REQUESTS_PATH=/cluster/work/igp_psr/spanwar/Open-d4rt/d4rt_agent/fixtures/qwen_grounding_tool_smoke.json,FORCE=0 \
  scripts/run_qwen_grounding_tool_debug_blackwell.slurm
```

If the directory already exists, choose a new name. Never delete or overwrite an earlier smoke run.

### 18.4 Smoke gate

Before the full DSI run:

- all raw outputs parse or produce a clearly recorded parser failure;
- bbox overlays are on the requested source frames;
- points are labeled and remain distinct;
- returned point count is at most requested count;
- at least the obvious-object bbox is visually plausible;
- no grounding prompt contains DSI context;
- only one physical Qwen checkpoint was loaded.

`not_found` is allowed but not required for the intentionally absent prompt; Qwen may still hallucinate. Record that behavior rather than tuning around one image.

If obvious-object grounding is grossly wrong, stop and inspect prompt/image ordering and coordinate conversion before running D4RT.

## 19. Prepare the fixed 25-question evaluation

### 19.1 Canonical references

The exact manifest is:

```text
d4rt_agent/results/dsi_bench/manifest.json
```

It contains:

- benchmark `DSI-Bench`;
- split `std`;
- 25 questions;
- requested sampling seed `20260721`;
- selected balanced seed `20260725`;
- five examples from each source dataset;
- the existing fixed question IDs.

Do not rebuild the manifest. Do not sample a new subset.

Reference results:

- previous committed D4RT agent: 10/25, 25 completed;
- unchanged Qwen-only baseline: 13/25;
- archived failed forced-grounding agent: 7/25, 21 completed.

The failed result is recoverable from:

```text
archive/failed-forced-grounding-prompt
```

### 19.2 Staging path

Use:

```text
/cluster/scratch/spanwar/tmp/d4rt_agent/phase_2_grounding_tool_staging
```

Create a fresh child directory. Never mix outputs from two Git commits.

Before the run, preserve comparison inputs under the staging parent:

- canonical manifest;
- previous 10/25 answer JSONs;
- unchanged baseline JSONs;
- previous report/findings;
- archived failed-run answer JSONs and findings.

Do not delete or overwrite an existing scratch run. Create a new child if a path is non-empty.

Prepare the staging and reference directories in one shell:

```bash
PHASE2_STAGE_PARENT="/cluster/scratch/spanwar/tmp/d4rt_agent/phase_2_grounding_tool_staging"
PHASE2_RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=12 HEAD)"
PHASE2_RUN_DIR="$PHASE2_STAGE_PARENT/run_$PHASE2_RUN_TAG"
PHASE2_REFERENCE_DIR="$PHASE2_STAGE_PARENT/references_$PHASE2_RUN_TAG"
test ! -e "$PHASE2_RUN_DIR"
test ! -e "$PHASE2_REFERENCE_DIR"
mkdir -p "$PHASE2_RUN_DIR/baseline"
mkdir -p "$PHASE2_REFERENCE_DIR/previous_agent/answers"
mkdir -p "$PHASE2_REFERENCE_DIR/previous_agent/baseline"
mkdir -p "$PHASE2_REFERENCE_DIR/failed_forced_grounding"
```

If either existence check fails, choose a new tag. Do not remove an existing run.

Copy the fixed manifest and unchanged baseline into staging:

```bash
cp -- d4rt_agent/results/dsi_bench/manifest.json "$PHASE2_RUN_DIR/manifest.json"
cp -a -- d4rt_agent/results/dsi_bench/baseline/. "$PHASE2_RUN_DIR/baseline/"
cmp -- d4rt_agent/results/dsi_bench/manifest.json "$PHASE2_RUN_DIR/manifest.json"
```

Preserve the previous committed comparison artifacts:

```bash
cp -- d4rt_agent/results/dsi_bench/manifest.json "$PHASE2_REFERENCE_DIR/previous_agent/manifest.json"
cp -a -- d4rt_agent/results/dsi_bench/answers/. "$PHASE2_REFERENCE_DIR/previous_agent/answers/"
cp -a -- d4rt_agent/results/dsi_bench/baseline/. "$PHASE2_REFERENCE_DIR/previous_agent/baseline/"
cp -- d4rt_agent/results/dsi_bench/DSI_BENCH_RESULTS.md "$PHASE2_REFERENCE_DIR/previous_agent/DSI_BENCH_RESULTS.md"
cp -- d4rt_agent/results/dsi_bench/FINDINGS.md "$PHASE2_REFERENCE_DIR/previous_agent/FINDINGS.md"
```

Export the archived failed run without switching branches:

```bash
PHASE2_FAILED_TAR="/tmp/d4rt_agent_failed_forced_grounding_$PHASE2_RUN_TAG.tar"
git archive \
  --format=tar \
  --output="$PHASE2_FAILED_TAR" \
  archive/failed-forced-grounding-prompt \
  d4rt_agent/results/dsi_bench/answers \
  d4rt_agent/results/dsi_bench/DSI_BENCH_RESULTS.md \
  d4rt_agent/results/dsi_bench/FINDINGS.md
tar -xf "$PHASE2_FAILED_TAR" -C "$PHASE2_REFERENCE_DIR/failed_forced_grounding"
```

Verify all three reference sets before submission:

```bash
find "$PHASE2_RUN_DIR/baseline" -maxdepth 1 -name '*.json' | wc -l
find "$PHASE2_REFERENCE_DIR/previous_agent/answers" -maxdepth 1 -name '*.json' | wc -l
find "$PHASE2_REFERENCE_DIR/failed_forced_grounding/d4rt_agent/results/dsi_bench/answers" -maxdepth 1 -name '*.json' | wc -l
sha256sum d4rt_agent/results/dsi_bench/manifest.json "$PHASE2_RUN_DIR/manifest.json"
```

Expected: `25`, `25`, `25`, and identical manifest hashes.

### 19.3 Record code identity

Commit all implementation and tests before the full evaluation.

Required pre-submit state:

```bash
git status --short
git rev-parse HEAD
```

Worktree must be clean. Record the commit SHA in:

- the SLURM log;
- a run metadata JSON;
- every answer record or a shared manifest referenced by every record.

Also record:

- Qwen model snapshot path;
- D4RT config/checkpoint;
- seed 42;
- main-agent token budget 768;
- grounder token budgets;
- max steps 14;
- point mode `ensemble5`;
- manifest SHA-256;
- orchestration and grounding prompt versions.

## 20. Run the full evaluation on Blackwell

Submit:

```bash
sbatch --parsable \
  --export=ALL,POINT_MODE=ensemble5,RESULTS_DIR="$PHASE2_RUN_DIR",MANIFEST_PATH="$PHASE2_RUN_DIR/manifest.json",FORCE=0 \
  scripts/run_dsi_bench_blackwell.slurm
```

Do not set `ONLY_FILE` or `LIMIT`; all 25 questions must run.
Keep this in the same shell where `PHASE2_RUN_DIR` was set.

Verify the scheduled resource:

```bash
scontrol show job <job-id>
```

It must show RTX PRO 6000/Blackwell constraints. If it is an A100 allocation, cancel before model execution and correct the request.

Monitor without busy-waiting:

```bash
squeue -j <job-id>
sstat -j <job-id>.batch
tail -n 100 logs/d4rt_agent/dsi_bench_<job-id>.out
```

The launcher writes one atomic JSON per question. If the four-hour job times out:

1. Do not use `FORCE=1`.
2. Do not change code or prompts.
3. Resubmit the exact same command and results directory.
4. Confirm completed IDs are skipped.
5. Continue until all 25 files exist.

If code or prompt changes after partial output:

- abandon that run directory as a recorded partial experiment;
- commit the fix;
- create a new empty run directory;
- rerun all 25 from scratch.

Never mix records generated by different commits.

## 21. Validate the staged results

Do not promote results immediately after the job exits.

### 21.1 File-level checks

Verify:

- exactly 25 answer JSON files;
- their filenames exactly match manifest question IDs;
- no extra answer JSON;
- every file parses;
- every file has current run metadata;
- every `created_at` is from this run;
- every record uses the same code SHA and manifest SHA.

### 21.2 Infrastructure versus legitimate agent failure

Classify statuses:

- `error` caused by model loading, CUDA, file I/O, schema bugs, or uncaught exceptions is an infrastructure failure. Fix and rerun; do not promote.
- `failed` after a valid 14-step agent attempt is an evaluation outcome. Audit it, record it, and count it incorrect.
- `complete_relaxed` is an evaluation outcome under the unchanged retry policy. Report it separately.
- `complete` is a strict completed answer.

There must be zero infrastructure errors before promotion.

### 21.3 Orchestration checks

Run the Phase 1 audit across all strict and any relaxed attempts:

- one effective action per turn;
- no fabricated host result in effective history;
- every ledger matches the registry;
- no phantom evidence ID;
- every rejected action creates no evidence;
- counters have no gaps except intentional cached qg reuse;
- raw suffixes remain diagnostic only.

### 21.4 Grounding provenance checks

For every accepted D4RT action:

- its `grounding_id` exists;
- it belongs to the same question/clip;
- status is `ok`;
- source frame matches the stored qg;
- bbox mode used host `ensemble5`;
- points mode uses only valid selected point IDs;
- no parsed D4RT action contains raw coordinates;
- D4RT evidence names the qg and point IDs.

For every grounding action:

- parsed coordinates satisfy bounds;
- point IDs are unique within qg;
- cache hits reuse an identical immutable record;
- raw grounder response is saved;
- overlay exists on exact `t_src`.

### 21.5 Result metrics

Implement `d4rt_agent/dsi_bench_compare.py` as a CPU-only command so these metrics are reproducible rather than hand-counted.

Required interface:

```text
python -m d4rt_agent.dsi_bench_compare \
  --current-results DIR \
  --previous-results DIR \
  --failed-results DIR \
  --output-json PATH \
  --output-markdown PATH
```

The command must read the current manifest as the authoritative question order and ground truth. It must parse the answer letter from final-answer text with the same choice parser used by the report, count failed/unparseable rows as incorrect, and fail if any comparison directory has missing or extra question IDs.

Compute:

- strict completion rate;
- relaxed completion rate;
- failed count;
- accuracy out of all 25, with failures incorrect;
- accuracy among completed answers;
- qg calls, actual Qwen grounding inferences, and cache hits;
- bbox versus points calls;
- `not_found` count;
- malformed grounder count;
- D4RT/math/rejected calls;
- steps per question;
- discarded-suffix frequency;
- D4RT visibility coverage by grounding mode;
- total and per-question runtime;
- question-level answer flips against:
  - previous 10/25 agent;
  - failed 7/25 forced-grounding agent;
  - unchanged 13/25 baseline.

Accuracy improvement is not a technical promotion gate. Structural correctness and reproducibility are.

Add unit tests with small synthetic manifests that cover completed, failed, relaxed, and unparseable answers. Before promotion, compare the generated headline counts against the known references: previous must be 10/25, failed must be 7/25, and baseline must be 13/25. If those checks disagree, stop and fix the input paths or parser before interpreting the new run.

## 22. Generate the complete staged `DSI_BENCH_RESULTS.md`, overlays, and audits

Do not generate or label a final report while the run is partial. First satisfy all staged-result checks in Section 21, including exactly 25 answer JSON files matching the fixed manifest.

Generate:

- `$PHASE2_RUN_DIR/DSI_BENCH_RESULTS.md` covering all 25 examples;
- per-question grounding sheets;
- per-frame combined qg overlays;
- orchestration audit;
- machine-readable summary metrics.

Run:

```bash
/cluster/work/igp_psr/spanwar/envs/d4rt_bw/bin/python -m d4rt_agent.dsi_bench_report \
  --results-dir "$PHASE2_RUN_DIR" \
  --refresh-groundings
```

Verify the deliverable immediately:

```bash
test -s "$PHASE2_RUN_DIR/DSI_BENCH_RESULTS.md"
rg -c '^## [0-9]+\.' "$PHASE2_RUN_DIR/DSI_BENCH_RESULTS.md"
```

Expected:

- the report exists and is non-empty;
- the question-section count is exactly `25`;
- the report headline states 25 questions;
- every manifest question appears once;
- every row uses the fresh staged agent result and unchanged baseline;
- failed or relaxed outcomes are shown honestly rather than omitted;
- links to grounding overlays resolve inside the staged results directory.

Treat a missing, partial, or stale `DSI_BENCH_RESULTS.md` as a failed Phase 2 deliverable. Fix report generation before promotion.

Inspect every grounding frame, not a hand-picked subset.

For each wrong or failed question, classify the dominant failure:

- wrong semantic request by main agent;
- bbox localization;
- part/point localization;
- ambiguous/static-background choice;
- D4RT visibility/tracking;
- coordinate-frame reasoning;
- trajectory calculation;
- option selection;
- orchestration/termination.

Do not call IoU against the old agent’s box “accuracy.” DSI-Bench has no ground-truth boxes.

## 23. Promote the verified run to canonical results

The user authorized replacing the canonical DSI result after the staged run is complete.

Before overwriting, ensure the previous 10/25 answers/report/findings are copied to the Phase 2 scratch reference directory and readable.

Promotion order:

1. Confirm staged validation has zero infrastructure errors.
2. Confirm exactly 25 staged answer files.
3. Copy the staged answer JSONs over the same 25 canonical filenames.
4. Leave the canonical manifest unchanged after verifying its hash matches staging.
5. Leave the unchanged baseline JSONs unchanged.
6. Run the report generator against canonical results with `--refresh-groundings`.
7. Regenerate the canonical `d4rt_agent/results/dsi_bench/DSI_BENCH_RESULTS.md` from all 25 promoted answers.
8. Write a new `FINDINGS.md` with the comparisons in Section 21.5.
9. Verify every per-frame grounding overlay and report link exists.
10. Inspect `git diff --stat` and `git status --short`.

The answer copy is intentionally an overwrite and is authorized only after the reference backup and staged checks above:

```bash
find "$PHASE2_RUN_DIR/answers" -maxdepth 1 -name '*.json' | wc -l
cp -a -- "$PHASE2_RUN_DIR/answers/." d4rt_agent/results/dsi_bench/answers/
find d4rt_agent/results/dsi_bench/answers -maxdepth 1 -name '*.json' | wc -l
```

Both counts must be 25. The staged validator must already have proved that the filename sets exactly match; line counts alone are not sufficient.

Do not promote:

- scratch logs;
- temporary smoke outputs;
- a rebuilt/different manifest;
- A100 outputs;
- stale old grounding overlays;
- partial records from another commit.

After copying the verified answers, use these commands:

```bash
/cluster/work/igp_psr/spanwar/envs/d4rt_bw/bin/python -m d4rt_agent.dsi_bench_report \
  --results-dir d4rt_agent/results/dsi_bench \
  --refresh-groundings
test -s d4rt_agent/results/dsi_bench/DSI_BENCH_RESULTS.md
rg -c '^## [0-9]+\.' d4rt_agent/results/dsi_bench/DSI_BENCH_RESULTS.md
```

The final command must print `25`. Do not commit the evaluation until the canonical report passes this check.

Then run the comparison command with:

- current: `d4rt_agent/results/dsi_bench`;
- previous: `$PHASE2_REFERENCE_DIR/previous_agent`;
- failed: `$PHASE2_REFERENCE_DIR/failed_forced_grounding/d4rt_agent/results/dsi_bench`;
- Markdown output: `d4rt_agent/results/dsi_bench/FINDINGS.md`;
- JSON output: the Phase 2 scratch run directory.

Inspect both generated files before committing. Do not blindly accept a report whose reference headline counts differ from 10/25, 7/25, and 13/25.

## 24. Final report requirements

`d4rt_agent/results/dsi_bench/FINDINGS.md` must state:

- exact implementation commit and Blackwell job ID(s);
- that the same fixed 25-example manifest was used;
- hardware: RTX PRO 6000 Blackwell;
- old agent 10/25;
- baseline 13/25;
- failed prompt-only run 7/25 with four failures;
- new accuracy and completion;
- strict versus relaxed outcomes;
- per-task results;
- question-level gains/losses;
- grounding mode/call/cache/not-found statistics;
- D4RT visibility statistics;
- runtime;
- orchestration audit result;
- representative overlay links;
- limitations and failure taxonomy;
- whether the hypothesis was supported, weakened, or unresolved.

If accuracy falls, say so directly. If grounding improves but reasoning does not, separate those findings. If orchestration is clean but performance is unchanged, that is still an informative result.

## 25. Commit strategy

Use these reviewable commits:

1. `feat(d4rt-agent): add isolated Qwen bbox and points grounding`
   - grounder prompt, parser, IDs/cache, tests.
2. `feat(d4rt-agent): query D4RT through immutable groundings`
   - new schemas, backend explicit-point path, provenance, tests.
3. `docs(d4rt-agent): teach grounding-tool workflows`
   - both system prompts and prompt tests.
4. `feat(d4rt-agent): report grounding provenance and overlays`
   - report/audit/smoke launcher changes and tests.
5. `eval(d4rt-agent): record grounding-tool DSI results`
   - 25 canonical answer records, refreshed overlays/report/findings.

Before each commit:

```bash
git diff --check
git status --short
git diff --cached --name-status
```

After the implementation commits and before the full run, the worktree must be clean. After result promotion, commit only the intended canonical result artifacts.

Do not push any commit.

## 26. Phase 2 completion criteria

Phase 2 is complete only when every item is true:

- [ ] Phase 1’s single-action and evidence-ledger invariants still pass.
- [ ] The main agent exposes exactly four tools.
- [ ] The main agent cannot emit D4RT coordinates.
- [ ] Grounder inference sees one frame and no DSI/history context.
- [ ] One physical Qwen checkpoint is reused sequentially.
- [ ] Bbox and points outputs are strictly validated.
- [ ] Null/empty results become explicit `not_found`.
- [ ] The host assigns immutable qg and point IDs.
- [ ] Identical requests reuse the same qg ID without another inference.
- [ ] D4RT accepts grounding IDs and validates clip/source provenance.
- [ ] Bbox qg uses host `ensemble5`.
- [ ] Point qg returns separate, unaveraged trajectories.
- [ ] CPU parser, contract, backend, isolation, cache, orchestration, and report tests pass.
- [ ] The grounding-only smoke test ran on RTX PRO 6000 Blackwell.
- [ ] The full run used the unchanged fixed 25-question manifest.
- [ ] The full run used RTX PRO 6000 Blackwell, seed 42, max 14 steps, and `ensemble5`.
- [ ] There are exactly 25 fresh staged answer records and zero infrastructure errors.
- [ ] The staged run generated a non-empty `DSI_BENCH_RESULTS.md` containing exactly 25 question sections.
- [ ] Orchestration and grounding provenance audits pass.
- [ ] Every grounding request has an exact-frame saved overlay.
- [ ] Verified results replaced the canonical 25 answer files.
- [ ] Canonical `d4rt_agent/results/dsi_bench/DSI_BENCH_RESULTS.md`, findings, and overlays were regenerated from all 25 promoted answers.
- [ ] Results are compared honestly with 10/25, 7/25, and 13/25 references.
- [ ] Scratch smoke/staging artifacts remain retrievable.
- [ ] No A100 job was used.
- [ ] Nothing was pushed.

## 27. Final design principle

The main Qwen decides what evidence it needs. The isolated Qwen localizes that request in one frame. The host freezes the geometry and D4RT measures it.

```text
reasoning request
→ isolated visual grounding
→ immutable qg ID
→ D4RT trajectory
→ restricted calculation
→ final answer
```
