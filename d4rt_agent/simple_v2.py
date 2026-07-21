"""Simple offline Qwen -> live D4RT agent (Phase 7).

Examples::

    python -m d4rt_agent.simple_v2 --point-mode centroid
    python -m d4rt_agent.simple_v2 --point-mode ensemble5

The selected point policy is immutable host configuration.  It is intentionally
absent from every schema and message shown to Qwen.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .simple_v2_backend import (
    DEFAULT_D4RT_CHECKPOINT,
    DEFAULT_D4RT_CONFIG,
    LiveD4RTBackend,
)
from .simple_v2_contracts import (
    ACTION_SCHEMAS,
    DEFAULT_ASSUMED_FPS,
    NUM_SAMPLED_FRAMES,
    POINT_MODES,
    SampledVideo,
    restricted_python_math,
    sample_video_cpu,
    validate_action,
)
from .simple_v2_eval import (
    DEFAULT_DEMO_DATA,
    DEFAULT_GT_SEED,
    DEFAULT_WORLDTRACK_NPZ,
    METRE_SCORED_TASKS,
    MatchedWorldTrackGT,
    aggregate_files,
    load_alignment_scale_from_metadata,
    score_question,
)


DEFAULT_DEMO_DIR = Path("demo/pstudio_mini/basketball_6")
DEFAULT_VIDEO = DEFAULT_DEMO_DIR / "assets" / "input_video.mp4"
DEFAULT_RESULTS_DIR = Path("d4rt_agent/results/basketball_6")
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
QUESTIONS: tuple[dict[str, str], ...] = (
    {
        "id": "endpoint_displacement",
        "question": (
            "What is the metric distance between the starting and ending position "
            "of the basketball?"
        ),
    },
    {
        "id": "distance_travelled",
        "question": (
            "How much distance did the basketball cover between the first and last "
            "frame of the video?"
        ),
    },
)


SYSTEM_PROMPT = """You are a 4D reasoning agent. You see a video and answer quantitative
questions about where objects are, how far they are, how far they moved, and how fast.
You are not a passive tool caller: you decide what physical quantity the question asks
for, decompose it into 3D facts you can measure, obtain those facts with D4RT, and
combine them with explicit arithmetic. The measurement is D4RT's job; the reasoning and
the plan are yours.

The video has exactly 32 uniformly sampled full-resolution RGB images, each explicitly
labelled Sampled frame 0 through Sampled frame 31. Pixel boxes use
[x_min,y_min,x_max,y_max] coordinates from 0 to 1000, with the origin at top-left.

WHAT D4RT IS
D4RT is a 3D point tracker over this exact 32-frame clip. You point at a physical point
on an object in one image; D4RT follows that same physical point through the clip and
reports where it is in 3D at every frame you ask about. It gives you what the images
alone cannot: metric depth and metric motion.

Concretely, one query_d4rt call means:
  "Here is a box around <label> as it appears in Sampled frame t_src. Tell me the 3D
   position of that object at sampled frames t_tgt, as seen from the camera at frame
   t_cam."
Inputs you must supply:
- label: what you are pointing at, in words.
- bbox_2d_1000: a tight box around that object *as it looks in frame t_src*, in
  [0,1000] coordinates. The host converts your box into the query point(s) it needs.
- t_src: the single frame you actually looked at to draw the box. This is a factual
  claim about your own grounding, not a preference.
- t_tgt: the list of frames whose 3D positions you want back. This list, and nothing
  you write in prose, decides what you receive.
- t_cam: the vantage point. Positions come back in the camera frame of this sampled
  frame, so t_cam answers "from where do you want to look at the scene". You choose it
  per query; it is not fixed and it need not be 0.

These four are independent. The object does not have to be visible in frame 0, you do
not have to ground it in frame 0, and you do not have to measure from frame 0. Ground
where the object is clearest (t_src), measure at the frames the question is about
(t_tgt), and view from the frame the question is asked from (t_cam).

What comes back, per requested target frame (in "predictions", same order as t_tgt):
- benchmark_aligned_xyz_m: [x,y,z] in meters, in the camera frame of t_cam. Use this
  for all metric answers. raw_xyz is unaligned model units; do not report it.
- visible: whether D4RT considers the tracked point observed at that frame. When false,
  the position is unreliable and benchmark_aligned_xyz_m may be null.
- confidence, target_uv_px: tracking confidence and where the point projects in that
  image, useful for sanity checks.
Plus two flat arrays built for calculations: math_trajectory_aligned_xyz_m (an [N,3]
list of positions, one row per t_tgt, in order) and math_visibility (an [N] boolean
mask). Occluded rows in the trajectory array are zero-filled, so always pass the
visibility mask alongside it and never treat a zero row as a real position.

Three things follow from this design and drive most of your planning:
1. One query = one object over time. To reason about two objects, issue one query per
   object, each grounded in a frame where that object is clearly visible, and give both
   queries the same t_cam so their positions land in one shared frame and can be
   subtracted directly.
2. You choose the time resolution. Two target frames give you endpoints; the full
   ordered list gives you a trajectory. Ask for what the quantity actually needs.
3. You choose the vantage point, and it changes what the coordinates mean.

THE COORDINATE FRAME
Inside one query, every returned position is expressed in the camera frame of t_cam,
using OpenCV axes: +x points right, +y points DOWN, +z points forward into the scene,
and the origin is that camera itself. So norm(position) is the distance from the camera
at frame t_cam, z is depth in front of it, and a smaller y is higher up in the world.

Choosing t_cam matters in two different ways, and you should keep them apart:
- For distances, it does not change the true answer. Moving the viewpoint is a rigid
  transform, and rigid transforms preserve distance, so a displacement or a path length
  measured entirely inside one query is the same number under any t_cam.
- For anything directional or egocentric, it changes the question. "How far is the
  cupboard from here" and "which way is the cupboard" are only well posed once "here"
  is named. If the question is asked from a particular moment -- "if we move to frame
  25, which direction is the cupboard" -- set t_cam to that frame and read the
  direction off the returned vector. Answering that from a different viewpoint answers
  a different question.
When the camera itself moves, this distinction is the whole game. When the camera is
static, all viewpoints nearly coincide and the choice barely matters.

Positions from two queries that used different t_cam are in different coordinate
systems. Subtracting them is meaningless, even though the arithmetic will succeed and
return a plausible-looking number. Whenever you intend to combine positions across
queries, give those queries the same t_cam. Each tool result restates its own frame in
its coordinate_frame field; check it before combining anything.

HOW TO TURN A QUESTION INTO MEASUREMENTS
Think in this order: which object(s), which frame(s), which viewpoint, which formula.
- Distance from the camera at some moment: set t_cam to that moment and query the
  object at that frame; the answer is norm(position), because the origin is that camera.
- Distance between two objects at some moment: query each object at that same frame
  under a shared t_cam; the answer is dist(a, b).
- Direction or bearing from a viewpoint: set t_cam to the frame the question is asked
  from and read the returned vector's components, remembering that +y is down.
- Displacement of one object between two moments: query it at both frames; the answer
  is dist(start, end).
- Total distance travelled: query every frame in the interval and sum consecutive
  steps with path_length(points, visibility).
- Speed: displacement divided by elapsed time. Pick a baseline of two frames far
  enough apart that the motion is well above tracking noise, take dist(start, end),
  and divide by the seconds between those two sampled frames, which the user message
  states. Average speed over a whole interval uses path length over elapsed time;
  straight-line speed uses displacement over elapsed time. Say which you computed.
- Size, gap, or separation questions reduce to distances between grounded points, so
  ground each end of the measurement separately.
- Direction of motion, approaching versus receding: compare norms or component
  differences across frames.
If a quantity is not directly returned, build it from positions and time. That
composition step is the part only you can do.

Prefer the fewest queries that fully support the number: one grounding per object,
with all the frames you need in a single t_tgt list.

ANSWER KINDS
final_answer carries a "kind" field that selects the answer shape. It defaults to
"numeric" when you omit it.
- "numeric" is the default and covers every question whose answer is one measured
  quantity. Report it in the unit the question names; if the question names none, use SI
  base units -- meters for distance, seconds for time, meters per second for speed,
  degrees for angles. A numeric answer must still cite the D4RT call and the python_math
  call behind it.
- "text" is for questions whose natural answer is words rather than one number. That
  covers questions that are not measurements at all, and it covers directional or
  qualitative answers such as "which way did it move" or "where is it relative to me".
  For a directional answer, still measure first with query_d4rt and python_math, then
  describe the direction from the components you measured: +x is to the right, -y is
  upward, and z is the distance ahead of the camera at t_cam. Cite the evidence you used.
Do not use "text" to dodge a measurement you could report as a number. A text answer is
recorded as unscored, so it is never a way to score better on a measurement question.

RESPONSE FORMAT
Think out loud in plain text before every action. State the quantity being measured,
the object or objects involved, the frames you need and why, and the formula that turns
the returned positions into the answer. Then condense that reasoning into the action's
justification field in one or two lines, so every tool call carries its own rationale.
The response must end with exactly one JSON action and nothing after it:
{"action":"ACTION_NAME","arguments":{...}}
Do not wrap it in markdown or code fences. Only the JSON is executed. Your reply is cut
off at a fixed token budget, so keep the reasoning to a few sentences; deliberating too
long truncates the action before it is emitted. Use only the three supplied schemas:
query_d4rt to measure, python_math to compute, final_answer to report.

TASK CLASSIFICATION
Decide the requested measurement and its time interval separately before choosing
targets:
- Endpoint displacement means straight-line separation between the first and last 3D
  positions of the requested interval. Typical wording includes "starting and ending
  position", "endpoint displacement", or "straight-line distance".
- Travelled path length means total distance accumulated along the route. Typical
  wording includes "distance covered", "how far did it travel", "travelled distance",
  "trajectory length", or "path length".
- Travel wording has priority: "distance covered between the first and last frame" is
  travelled path length, not endpoint displacement. The endpoints define the interval,
  not the measurement.

TIME INTERVAL
- If the user names sampled frames A and B, use A and B as the inclusive interval.
- "First and last frame of the video" means sampled frames 0 and 31.
- "First and last appearance" means the earliest and latest labelled sampled frames
  where the requested object is visually present. Determine those frames from the
  images; do not substitute video frames 0 and 31 unless the object appears there.
- If the user gives no temporal bounds, use the object's first and last visible
  appearance, not automatically the first and last video frame.
An object may disappear and later reappear inside the interval. That creates disjoint
visible segments; it does not reduce a path request to two endpoints.

GROUNDING AND QUERY CONTRACT
Use query_d4rt for all 3D facts. Every query must declare exactly one
t_src: the sampled frame in which you visually grounded bbox_2d_1000. Derive the box
from that labelled source image, make it tight around the requested object, and do not
reuse a box with a different t_src unless you independently grounded it in that image.
Prefer one source-frame grounding with all required targets in one t_tgt list.

The actual t_tgt JSON array determines which positions D4RT returns; claims made only
in the justification have no effect. After selecting inclusive interval [A,B]:
- Endpoint displacement: query A and B. These are not necessarily 0 and 31.
- Travelled path length: query every sampled frame A,A+1,...,B in order. For the full
  video interval this is
  "t_tgt":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31].
  [0,31] alone is never a path over the full video and can measure only endpoint
  displacement.
For travelled path, pass the complete ordered trajectory and returned visibility mask
to path_length. It sums consecutive steps within each visible segment, resets at every
invisible frame, and never bridges a disappearance/reappearance gap. Travel while the
object is absent is unobserved and must be stated as a limitation. For example, if the
object is visible on frames 0-10 and 20-31, the result is path(0-10) + path(20-31),
never a step from frame 10 to frame 20. If an action is
rejected for incomplete path evidence, change the next query's actual t_tgt array,
not merely its justification. Never repeat rejected arguments.

CALCULATION CONTRACT
python_math is where measured positions become the answer. It cannot see the video or
call D4RT; it only evaluates arithmetic over values you pull out of earlier tool
results. Each binding names a variable and points at a field of one recorded evidence
object, in this exact form:
{"variable":{"evidence_id":"d4rt_1","path":["field",0,"subfield"]}}
The path walks the tool result: string components index object fields, integers index
lists. Bindings may draw on several different evidence IDs in one call, which is how
you compare two objects: bind one position from d4rt_1 and the other from d4rt_2, then
take dist between them. Numeric constants such as elapsed seconds may be written
directly in the code.
Useful safe functions include dist(a,b), norm(a), path_length(points,visibility),
mean(a), std(a), sqrt(x), abs(x), min(a), max(a), sum(a), and round(x,n).
For a full trajectory, bind each field to a separate numeric variable, for example:
{"points":{"evidence_id":"d4rt_1","path":["math_trajectory_aligned_xyz_m"]},
 "visible":{"evidence_id":"d4rt_1","path":["math_visibility"]}}
Then use code such as "value = path_length(points, visible)". Do not use string-key
indexing inside code; bind each field you need to its own variable first. Metric
answers must use benchmark-aligned meters.

Before final_answer, call python_math. The final evidence_ids must cite both the live
D4RT call and the python_math call used for the number. State visibility or sparse
point limitations briefly.

PLANNING AND TOOL-USE EXAMPLES
The labels, boxes, evidence values, and final numbers below are illustrative. Never
copy them into a real answer; visually identify the requested object and derive its
box from the declared source image.

Example A -- endpoint displacement
Question: What is the straight-line displacement of a red suitcase from Sampled frame
5 to Sampled frame 24?
Reasoning aloud: the quantity is straight-line displacement of one object, the red
suitcase, over the interval [5,24]; two positions suffice, so I ground the suitcase in
Sampled frame 5 and request targets [5,24] in one query; the answer is
dist(position_at_5, position_at_24); then cite both calls.
First action example:
{"action":"query_d4rt","arguments":{"label":"red suitcase","bbox_2d_1000":[620,430,710,610],"t_src":5,"t_tgt":[5,24],"t_cam":0,"justification":"Ground the requested object in Sampled frame 5 and obtain the user-specified endpoint positions."}}
After tool result d4rt_1, calculation example:
{"action":"python_math","arguments":{"bindings":{"start":{"evidence_id":"d4rt_1","path":["predictions",0,"benchmark_aligned_xyz_m"]},"end":{"evidence_id":"d4rt_1","path":["predictions",1,"benchmark_aligned_xyz_m"]}},"code":"value = dist(start, end)","justification":"Calculate endpoint displacement from the two aligned positions."}}

Example B -- travelled path length
Question: How much distance did a toy vehicle cover between the first and last frame?
Reasoning aloud: "distance covered" is travelled path length, not displacement, so two
endpoints are not enough; the interval is the whole video, so I ground the toy vehicle
in one declared source frame and request all 32 targets in a single query; the answer
is path_length over the returned trajectory, respecting the visibility mask so that
disappearance gaps are never bridged; then cite both calls.
First action example:
{"action":"query_d4rt","arguments":{"label":"toy vehicle","bbox_2d_1000":[120,680,260,820],"t_src":0,"t_tgt":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],"t_cam":0,"justification":"Ground the requested object in Sampled frame 0 and obtain every sampled position for travelled path length."}}
After tool result d4rt_1, calculation example:
{"action":"python_math","arguments":{"bindings":{"points":{"evidence_id":"d4rt_1","path":["math_trajectory_aligned_xyz_m"]},"visible":{"evidence_id":"d4rt_1","path":["math_visibility"]}},"code":"value = path_length(points, visible)","justification":"Sum each disjoint visible segment without bridging disappearance gaps."}}
If math_1 reports outputs.value=1.25, final action structure example:
{"action":"final_answer","arguments":{"kind":"numeric","value":1.25,"unit":"meters","evidence_ids":["d4rt_1","math_1"],"limitations":"Visibility-aware sum of observed segments; travel while absent is unobserved."}}

Example C -- composing a quantity D4RT does not return directly
Question: How fast is the cyclist moving?
Reasoning aloud: speed is not returned by D4RT; it is displacement over elapsed time.
The cyclist moves steadily, so I take a wide baseline, frames 4 and 28, to keep the
motion well above tracking noise, ground the cyclist in Sampled frame 4, and request
targets [4,28]. The user message gives the seconds per sampled step, so the elapsed
time is that step duration times 24. The answer is dist(start, end) divided by that
elapsed time, reported in meters per second.
Calculation example, where 4.8 stands for the elapsed seconds derived from the stated
timing:
{"action":"python_math","arguments":{"bindings":{"start":{"evidence_id":"d4rt_1","path":["predictions",0,"benchmark_aligned_xyz_m"]},"end":{"evidence_id":"d4rt_1","path":["predictions",1,"benchmark_aligned_xyz_m"]}},"code":"displacement = dist(start, end)\\nvalue = displacement / 4.8","justification":"Divide the baseline displacement by the elapsed time between the two sampled frames to obtain average speed."}}

The same shape covers two-object questions: query each object separately under the same
t_cam, then bind one position from each evidence ID and take dist between them.

Example D -- choosing the viewpoint
Question: If we move to Sampled frame 25, which direction is the cupboard?
Reasoning aloud: this is a directional question asked from a specific moment, so the
vantage point is part of the question and t_cam must be 25; answering from any other
viewpoint answers a different question. The cupboard is clearest in Sampled frame 20,
so I ground it there with t_src=20 and request t_tgt=[25] with t_cam=25. The returned
vector is the cupboard's offset from the frame-25 camera in OpenCV axes, so a positive
x means to the right, a negative y means above eye level, and z is how far ahead.
Query example:
{"action":"query_d4rt","arguments":{"label":"cupboard","bbox_2d_1000":[300,240,470,640],"t_src":20,"t_tgt":[25],"t_cam":25,"justification":"Ground the cupboard where it is clearest and express its position from the frame-25 viewpoint the question asks about."}}
"Which direction" is answered in words, not as one number, so after reading the measured
components the final action example is:
{"action":"final_answer","arguments":{"kind":"text","text":"From sampled frame 25 the cupboard is about 2.4 m ahead and roughly 1.1 m to the right, sitting a little above eye level.","evidence_ids":["d4rt_1","math_1"],"limitations":"Direction read from a single grounded point; the cupboard's extent is not measured."}}
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_action_json(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object representing an action."""

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and ("action" in value or "name" in value):
            return value
    raise ValueError(f"Qwen did not emit one JSON action: {text[:500]!r}")


def _redact_host_policy(result: dict[str, Any]) -> dict[str, Any]:
    """Remove host policy metadata before returning a tool result to Qwen."""

    redacted = {
        key: value for key, value in result.items() if key not in {"point_mode", "query_points"}
    }
    predictions = redacted.get("predictions")
    if isinstance(predictions, list):
        redacted["predictions"] = [
            {
                key: value
                for key, value in prediction.items()
                if key != "individual_point_results"
            }
            for prediction in predictions
        ]
    return redacted


def _resolve_path(value: Any, path: Sequence[Any]) -> Any:
    current = value
    for component in path:
        if isinstance(current, Mapping) and isinstance(component, str):
            if component not in current:
                raise ValueError(f"binding path field not found: {component!r}")
            current = current[component]
        elif isinstance(current, (list, tuple)) and isinstance(component, int):
            try:
                current = current[component]
            except IndexError as error:
                raise ValueError(f"binding path index out of range: {component}") from error
        else:
            raise ValueError(f"cannot apply binding path component {component!r}")
    return current


def resolve_bindings(
    specifications: Mapping[str, Any], evidence: Mapping[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve Qwen binding specifications only from recorded evidence."""

    values: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for variable, specification in specifications.items():
        if not isinstance(variable, str) or not variable.isidentifier() or variable.startswith("_"):
            raise ValueError(f"invalid binding variable: {variable!r}")
        if not isinstance(specification, Mapping):
            raise ValueError(f"binding {variable!r} must be an evidence reference object")
        evidence_id = specification.get("evidence_id")
        path = specification.get("path", [])
        if not isinstance(evidence_id, str) or evidence_id not in evidence:
            raise ValueError(f"binding {variable!r} cites unknown evidence: {evidence_id!r}")
        if not isinstance(path, list) or not all(isinstance(item, (str, int)) for item in path):
            raise ValueError(f"binding {variable!r} path must contain string fields or integer indices")
        values[variable] = _resolve_path(evidence[evidence_id], path)
        provenance[variable] = {"evidence_id": evidence_id, "path": path}
    if not values:
        raise ValueError("python_math requires at least one prior-evidence binding")
    return values, provenance


def _all_finite_scalars(value: Any) -> list[float]:
    found: list[float] = []
    if isinstance(value, Mapping):
        for child in value.values():
            found.extend(_all_finite_scalars(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(_all_finite_scalars(child))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        found.append(float(value))
    return found


def _validate_final_evidence(
    task_id: str,
    arguments: dict[str, Any],
    evidence: Mapping[str, dict[str, Any]],
) -> None:
    evidence_ids = arguments["evidence_ids"]
    if any(item not in evidence for item in evidence_ids):
        unknown = [item for item in evidence_ids if item not in evidence]
        raise ValueError(f"final answer cites unknown evidence IDs: {unknown}")
    if arguments.get("kind", "numeric") == "text":
        # Descriptive answers carry no scored number, so there is nothing to trace
        # back to a calculation.  They are recorded as unscored instead.
        return
    d4rt_ids = [item for item in evidence_ids if item.startswith("d4rt_")]
    math_ids = [item for item in evidence_ids if item.startswith("math_")]
    if not d4rt_ids or not math_ids:
        raise ValueError("final answer must cite both a d4rt_* and math_* call")
    targets = {
        int(value)
        for evidence_id in d4rt_ids
        for value in evidence[evidence_id]["t_tgt"]
    }
    if len(targets) < 2:
        raise ValueError("measurement evidence must query at least two sampled frames")
    interval_start, interval_end = min(targets), max(targets)
    if task_id == "endpoint_displacement":
        # The question determines the endpoints.  They may be explicit sampled
        # frames or the object's visually determined first/last appearance.
        pass
    elif task_id == "distance_travelled":
        expected = set(range(interval_start, interval_end + 1))
        if targets != expected:
            missing = sorted(expected - targets)
            raise ValueError(
                "path evidence must query every sampled frame in the selected "
                f"inclusive interval [{interval_start},{interval_end}]; missing {missing}"
            )
    # Any other task id simply has no bespoke evidence rule yet; the two-frame
    # minimum above still applies.  Task ids are host-supplied, never model-supplied,
    # so falling back here cannot be used to weaken a rule the model was given.
    candidates: list[float] = []
    for evidence_id in math_ids:
        candidates.extend(_all_finite_scalars(evidence[evidence_id].get("outputs", {})))
    value = float(arguments["value"])
    tolerance = max(1e-3, abs(value) * 1e-3)
    if not any(abs(value - candidate) <= tolerance for candidate in candidates):
        raise ValueError("final answer value does not match a cited calculation output")
    # Only the GT-scored tasks are pinned to meters, because their answer is compared
    # against a metres ground truth where "centimeters" would be a silent 100x error.
    if task_id in METRE_SCORED_TASKS:
        if arguments["unit"].strip().lower() not in {"m", "meter", "meters", "metre", "metres"}:
            raise ValueError(f"{task_id} answers are scored in meters and must use meters")


class OfflineQwen:
    """Deterministic local-only Qwen3-VL generator."""

    def __init__(self, model_id: str, max_new_tokens: int = 768, seed: int = 42) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.seed = int(seed)
        supplied = Path(model_id).expanduser()
        if supplied.exists():
            model_path = supplied.resolve()
        else:
            model_path = Path(
                snapshot_download(repo_id=model_id, local_files_only=True)
            ).resolve()
        self.model_path = str(model_path)
        self.device_map_policy = os.environ.get("QWEN_DEVICE_MAP", "auto").strip().lower()
        if self.device_map_policy == "auto":
            device_map: str | dict[str, int] = "auto"
        elif self.device_map_policy == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("QWEN_DEVICE_MAP=cuda requires an available CUDA device")
            device_map = {"": 0}
        else:
            raise ValueError("QWEN_DEVICE_MAP must be 'auto' or 'cuda'")
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="sdpa",
            local_files_only=True,
        ).eval()
        self.max_new_tokens = int(max_new_tokens)

    def generate(self, messages: list[dict[str, Any]]) -> str:
        self.torch.manual_seed(self.seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(self.seed)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        new_tokens = output[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


class OrchestrationError(RuntimeError):
    """Agent exhaustion that retains all replayable partial state."""

    def __init__(self, message: str, trace: list[dict[str, Any]], evidence: dict[str, dict[str, Any]]) -> None:
        super().__init__(message)
        self.trace = trace
        self.evidence = evidence


class SimpleV2Orchestrator:
    """Host-enforced three-action loop with replayable evidence."""

    def __init__(
        self,
        *,
        qwen: OfflineQwen,
        backend: LiveD4RTBackend,
        sampled_video: SampledVideo,
        max_steps: int = 12,
    ) -> None:
        self.qwen = qwen
        self.backend = backend
        self.sampled_video = sampled_video
        self.max_steps = max(1, int(max_steps))

    def solve(self, task: dict[str, str]) -> dict[str, Any]:
        from PIL import Image

        tool_text = json.dumps(ACTION_SCHEMAS, separators=(",", ":"))
        initial_content: list[dict[str, Any]] = []
        for index, frame in enumerate(self.sampled_video.frames_rgb):
            initial_content.append({"type": "text", "text": f"Sampled frame {index}:"})
            initial_content.append({"type": "image", "image": Image.fromarray(frame)})
        fps = float(self.sampled_video.fps)
        if not math.isfinite(fps) or fps <= 0:
            fps = DEFAULT_ASSUMED_FPS
        step_seconds = (
            self.sampled_video.original_indices[1] - self.sampled_video.original_indices[0]
        ) / fps
        initial_content.append({
            "type": "text",
            "text": (
                "Each image above is full resolution. Ground bbox_2d_1000 in the "
                "sampled frame declared by t_src.\n"
                f"Timing: consecutive sampled frames are {step_seconds:.4f} seconds "
                "apart, so the elapsed time between sampled frames A and B is "
                f"{step_seconds:.4f} * (B - A) seconds, and the whole clip spans "
                f"{step_seconds * 31:.4f} seconds.\n"
                f"Available action schemas: {tool_text}\n\n"
                f"Question: {task['question']}"
            ),
        })
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": initial_content},
        ]
        trace: list[dict[str, Any]] = []
        evidence: dict[str, dict[str, Any]] = {}
        counters = {"d4rt": 0, "math": 0, "final": 0}

        for step in range(1, self.max_steps + 1):
            raw = self.qwen.generate(messages)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": raw}]})
            attempt: dict[str, Any] = {
                "step": step,
                "kind": "model_action",
                "raw_qwen_response": raw,
            }
            try:
                if re.search(r"\bpoint_mode\b", raw):
                    raise ValueError("Qwen must never select or emit point_mode")
                parsed = _extract_action_json(raw)
                name, arguments = validate_action(parsed)
                attempt["parsed_action"] = {"action": name, "arguments": arguments}
                if name == "final_answer":
                    _validate_final_evidence(task["id"], arguments, evidence)
                    counters["final"] += 1
                    call_id = f"final_{counters['final']}"
                    record = {
                        **attempt,
                        "call_id": call_id,
                        "status": "ok",
                        "result": arguments,
                    }
                    trace.append(record)
                    return {
                        "point_mode": self.backend.point_mode,
                        **task,
                        "final_answer": arguments,
                        "trace": trace,
                        "evidence": evidence,
                        "steps": step,
                    }
                result, response_content, prefix = self._execute_action(
                    name, arguments, evidence
                )
                counters[prefix] += 1
                call_id = f"{prefix}_{counters[prefix]}"
                evidence[call_id] = result
                record = {
                    **attempt,
                    "call_id": call_id,
                    "status": "ok",
                    "justification": arguments["justification"],
                    "result": result,
                }
                trace.append(record)
                if response_content is None:
                    response_content = [{
                        "type": "text",
                        "text": f"Tool result {call_id}: {json.dumps(_redact_host_policy(result))}",
                    }]
                else:
                    response_content.insert(0, {
                        "type": "text",
                        "text": f"Tool result {call_id}: {json.dumps(_redact_host_policy(result))}",
                    })
                messages.append({"role": "user", "content": response_content})
            except (KeyError, TypeError, ValueError) as error:
                attempt.update(status="rejected", error=str(error))
                trace.append(attempt)
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": (
                            f"Action rejected: {error}. Return one corrected JSON action using "
                            "only the supplied schemas."
                        ),
                    }],
                })
        raise OrchestrationError(
            f"Qwen did not produce a valid final answer in {self.max_steps} steps; "
            f"last trace entry: {trace[-1] if trace else 'none'}",
            trace,
            evidence,
        )

    def _execute_action(
        self,
        name: str,
        arguments: dict[str, Any],
        evidence: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]] | None, str]:
        if name == "query_d4rt":
            result = self.backend.query(
                label=arguments["label"],
                bbox_2d_1000=arguments["bbox_2d_1000"],
                t_src=arguments["t_src"],
                t_tgt=arguments["t_tgt"],
                t_cam=arguments["t_cam"],
            )
            return result, None, "d4rt"
        if name == "python_math":
            values, provenance = resolve_bindings(arguments["bindings"], evidence)
            outputs = restricted_python_math(values, arguments["code"])
            result = {
                "bindings": provenance,
                "resolved_bindings": values,
                "code": arguments["code"],
                "outputs": outputs,
            }
            return result, None, "math"
        raise ValueError(f"host cannot execute action: {name}")


def replay_tool_trace(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate trace structure and replay every restricted calculation on CPU."""

    evidence: dict[str, dict[str, Any]] = {}
    replayed_math = 0
    final_answers = 0
    for entry in trace:
        if entry.get("status") != "ok":
            continue
        call_id = entry.get("call_id")
        if not isinstance(call_id, str):
            raise ValueError("successful trace entry is missing call_id")
        result = entry.get("result")
        if call_id.startswith("math_"):
            if not isinstance(result, Mapping):
                raise ValueError(f"invalid calculation result in {call_id}")
            replayed = restricted_python_math(result["resolved_bindings"], result["code"])
            if replayed != result["outputs"]:
                raise ValueError(f"calculation replay mismatch in {call_id}")
            replayed_math += 1
        if call_id.startswith("final_"):
            final_answers += 1
        if call_id.startswith(("d4rt_", "math_")):
            evidence[call_id] = dict(result)
    return {
        "status": "complete" if final_answers == 1 else "incomplete",
        "successful_evidence_calls": len(evidence),
        "replayed_math_calls": replayed_math,
        "final_answers": final_answers,
    }


def run_live(args: argparse.Namespace) -> dict[str, Any]:
    sampled = sample_video_cpu(args.video)
    scale, scale_provenance = load_alignment_scale_from_metadata(args.demo_data)
    gt = MatchedWorldTrackGT(
        args.gt_npz,
        sampled.original_indices,
        seed=(args.gt_seed_u, args.gt_seed_v, args.gt_seed_frame),
    )
    backend = LiveD4RTBackend(
        sampled_video=sampled,
        point_mode=args.point_mode,
        benchmark_scale=scale,
        model_config=args.d4rt_config,
        checkpoint=args.d4rt_checkpoint,
        device=args.device,
        dtype=args.d4rt_dtype,
        query_chunk_size=args.query_chunk_size,
    )
    if args.smoke:
        result = backend.query(
            label="smoke-test object",
            bbox_2d_1000=[520.0, 360.0, 580.0, 480.0],
            t_src=0,
            t_tgt=[31],
            t_cam=0,
        )
        return {
            "phase": "simple_v2_live_smoke",
            "created_at": _utc_now(),
            "point_mode": args.point_mode,
            "sampling": _sampling_record(sampled),
            "scale_provenance": scale_provenance,
            "d4rt": backend.metadata(),
            "query": result,
            "gpu": backend.gpu_memory(),
        }

    qwen = OfflineQwen(args.qwen_model, args.max_new_tokens, args.seed)
    orchestrator = SimpleV2Orchestrator(
        qwen=qwen,
        backend=backend,
        sampled_video=sampled,
        max_steps=args.max_steps,
    )
    question_results = []
    scores = []
    failures = []
    for task in QUESTIONS:
        try:
            solved = orchestrator.solve(dict(task))
        except OrchestrationError as error:
            solved = {
                "point_mode": args.point_mode,
                **task,
                "status": "failed",
                "error": str(error),
                "trace": error.trace,
                "evidence": error.evidence,
            }
            question_results.append(solved)
            failures.append({"task_id": task["id"], "error": str(error)})
            break
        # Record the solved question before scoring: a scoring error must never
        # discard an expensive completed run.
        question_results.append(solved)
        try:
            score = score_question(
                task_id=task["id"],
                final_answer=solved["final_answer"],
                evidence=solved["evidence"],
                gt=gt,
                width=sampled.width,
                height=sampled.height,
            )
        except Exception as error:  # noqa: BLE001 - never lose the artifact to scoring
            failures.append({
                "task_id": task["id"],
                "stage": "scoring",
                "error": f"{type(error).__name__}: {error}",
            })
            continue
        if score.get("scored", True):
            score["agent_vs_recomputed_aligned_delta_m"] = abs(
                float(score["agent_final_value"]) - float(score["benchmark_aligned_value_m"])
            )
        scores.append(score)

    return {
        "phase": "simple_v2_live_agent",
        "status": "failed" if failures else "complete",
        "created_at": _utc_now(),
        "point_mode": args.point_mode,
        "offline": True,
        "configuration": {
            "seed": args.seed,
            "qwen_decoding": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": args.max_new_tokens,
            },
            "point_mode_selected_by": "CLI or POINT_MODE environment before execution",
            "point_mode_exposed_to_qwen": False,
            "t_cam_selected_by": "Qwen, per query, over the full sampled range",
        },
        "sampling": _sampling_record(sampled),
        "qwen": {
            "model": qwen.model_path,
            "device_map": qwen.device_map_policy,
            "action_schemas": list(ACTION_SCHEMAS),
        },
        "d4rt": backend.metadata(),
        "scale_provenance": scale_provenance,
        "ground_truth": gt.canonical_trajectory(),
        "questions": question_results,
        "scores": scores,
        "failures": failures,
        "gpu": backend.gpu_memory(),
        "trace_replay": [
            replay_tool_trace(item.get("trace", [])) for item in question_results
        ],
    }


def _sampling_record(sampled: SampledVideo) -> dict[str, Any]:
    return {
        "contract": "round(linspace(0, N - 1, 32))",
        "decoded_on": "CPU",
        "video": str(sampled.video_path),
        "total_original_frames": sampled.total_original_frames,
        "num_sampled_frames": NUM_SAMPLED_FRAMES,
        "sampled_to_original": sampled.mapping(),
        "fps": sampled.fps,
        "width": sampled.width,
        "height": sampled.height,
        "same_frames_for_qwen_and_d4rt": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--point-mode",
        choices=POINT_MODES,
        default=os.environ.get("POINT_MODE", "centroid"),
        help="immutable host grounding policy (never included in the Qwen schema)",
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--qwen-model", default=os.environ.get("MODEL_DIR", DEFAULT_QWEN_MODEL))
    parser.add_argument("--d4rt-config", type=Path, default=DEFAULT_D4RT_CONFIG)
    parser.add_argument("--d4rt-checkpoint", type=Path, default=DEFAULT_D4RT_CHECKPOINT)
    parser.add_argument("--d4rt-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-data", type=Path, default=DEFAULT_DEMO_DATA)
    parser.add_argument("--gt-npz", type=Path, default=DEFAULT_WORLDTRACK_NPZ)
    parser.add_argument("--gt-seed-u", type=float, default=DEFAULT_GT_SEED[0])
    parser.add_argument("--gt-seed-v", type=float, default=DEFAULT_GT_SEED[1])
    parser.add_argument("--gt-seed-frame", type=int, default=DEFAULT_GT_SEED[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="run one live D4RT query without Qwen")
    parser.add_argument("--replay-trace", type=Path, help="CPU-validate traces in an existing result")
    parser.add_argument("--aggregate", action="store_true", help="aggregate completed centroid and ensemble artifacts")
    parser.add_argument(
        "--centroid-result", type=Path, default=DEFAULT_RESULTS_DIR / "simple_v2_centroid.json"
    )
    parser.add_argument(
        "--ensemble-result", type=Path, default=DEFAULT_RESULTS_DIR / "simple_v2_ensemble5.json"
    )
    parser.add_argument(
        "--aggregate-output", type=Path, default=DEFAULT_RESULTS_DIR / "simple_v2_aggregate.json"
    )
    parser.add_argument(
        "--aggregate-report", type=Path, default=DEFAULT_RESULTS_DIR / "SIMPLE_V2_RESULTS.md"
    )
    args = parser.parse_args(argv)
    if args.point_mode not in POINT_MODES:
        parser.error(f"unsupported POINT_MODE default: {args.point_mode!r}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.aggregate:
        result = aggregate_files(
            args.centroid_result,
            args.ensemble_result,
            args.aggregate_output,
            args.aggregate_report,
        )
        print(json.dumps(result["comparison"], indent=2))
        print(f"Saved: {args.aggregate_output}")
        return
    if args.replay_trace is not None:
        artifact = json.loads(args.replay_trace.read_text())
        reports = [replay_tool_trace(item["trace"]) for item in artifact["questions"]]
        print(json.dumps(reports, indent=2))
        if artifact.get("status") != "complete" or any(
            report["status"] != "complete" for report in reports
        ):
            raise SystemExit(1)
        return
    result = run_live(args)
    output = args.output
    if output is None:
        suffix = "_smoke" if args.smoke else ""
        output = DEFAULT_RESULTS_DIR / f"simple_v2_{args.point_mode}{suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if "scores" in result:
        for score in result["scores"]:
            task = score.get("task_id", "?")
            if not score.get("scored", True):
                answer = score.get("agent_final_text") or score.get("agent_final_value")
                print(
                    f"{task}: UNSCORED ({score.get('unscored_reason', 'no reason given')}) "
                    f"answer={answer!r}"
                )
                continue
            print(
                f"{task}: raw={score['raw_d4rt_value']:.4f} "
                f"aligned={score['benchmark_aligned_value_m']:.4f} m "
                f"GT={score['gt_value_m']:.4f} m error={score['absolute_error_m']:.4f} m"
            )
    print(f"Saved: {output}")
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
