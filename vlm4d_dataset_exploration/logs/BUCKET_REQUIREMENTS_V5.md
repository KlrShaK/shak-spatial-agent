# What AgentV5 needs for VLM4D, kind by kind

Hand analysis, not generated. Written in the same form as `dsi_dataset_exploration/logs/BUCKET_REQUIREMENTS.md`
and `dyn_dataset_exploration/logs/BUCKET_REQUIREMENTS_V5.md`: *Understanding needed / Sub-tasks / Inputs /
Outputs / Where AgentV5 stands*. Counts and accuracies come from `TAXONOMY_REPORT.md`. "Text prior" is the
video-blind baseline of §6 there; "7B" is Qwen2.5-VL-7B and "best" the best of the paper's 23 graded models
(chain-of-thought runs). The sub-task programs are `taxonomy.SUBTASKS`; each question's own program is in
`manifest.json`.

**Summary.** 1317 of 1816 questions (73%) land in kinds AgentV5 already has. 985 of them are one kind,
`subject_direction`. 490 (27%) need **six new kinds**, and 9 fit no kind:

| new kind | family | n | text prior | 7B | best | what is missing |
|---|---|---|---|---|---|---|
| `repetition_count` | count | 145 | 42.8 | 38.6 | 51.0 | an event series, not a first onset |
| `count_with_predicate` | count | 112 | 30.4 | 36.6 | 52.7 | a per-instance measurement before counting |
| `rotation_sense` | subject_motion | 108 | 53.7 | 50.9 | 60.2 | the sense of a spin, in the image plane or about the vertical |
| `relative_position` | two_entities | 55 | 56.4 | 47.3 | 67.3 | which side of B entity A is on |
| `subject_turn` | subject_motion | 41 | 53.7 | 22.0 | 56.1 | heading change without an own frame |
| `hand_attribution` | semantic | 29 | 27.6 | 55.2 | 69.0 | left/right hand grounding |
| (`other`: identity by motion, speed comparison) | -- | 9 | 55.6 | 33.3 | 77.8 | a motion filter over all objects |

Two cross-cutting steps matter more than any single kind: the **existence check** (795 questions offer an
absence option) and **actor resolution** for the Ego4D wearer (113 questions).

---

## 0. Cross-cutting steps (they come before the kind's own steps)

### 0a. `exists(named object)`: the false-positive option

**Understanding needed:** whether the object the *option* names is in the clip at all. The option's object
is not always the question's ("there are no buses in the video" on a kayak question; 21 such options).

**Inputs:** SAM3 grounding of the option's noun over the clip's frames, not just one prompt frame. Also
Qwen's yes/no on keyframes, because SAM3 will find *something* for "dragon" in a synthetic fantasy scene.

**Outputs:** a verdict with evidence: the best mask score over the frames, the number of frames with a mask,
and Qwen's answer. Calibrate on the base rates. The absence option is offered in 795 questions and is the
answer in 92 (12%; synthetic 45/445 = 10%). A detector that says "absent" more than about 1 time in 8
loses accuracy.

**Where AgentV5 stands:** `ground` returns an instance count. When it is 0, `_not_found` tells the model to
"try `ground` with another phrase", which treats absence as a tool failure to retry rather than a possible
answer. Nothing carries "the segmenter never found X" to the answer turn as evidence for the "no X" option.

### 0b. `resolve_actor(person = camera wearer)`

**Understanding needed:** in an Ego4D clip, "which way does the person turn / move / look" asks about the
camera. 113 questions; the rule `taxonomy._ego_wearer` flags them. "The person who clapped", "the
cyclist" and "the runner" are other people, and the rule leaves them alone.

**Where AgentV5 stands:** no notion of ego video. The family menu describes `camera` as "only the camera
itself", and Qwen is not told the clip is ego-centric, so it will choose `subject_motion` and try to ground
"person". The loader knows the source (`group == "ego4d"`); passing that to the D1 prompt is the
cheap fix.

### 0c. `window(start|end)`

75 questions name a phase: "at the end of the video", "at first", "after landing". There are no
timestamps. `program.window_from_question` already maps these phrases; nothing new is needed.

---

## 1. `subject_direction` (existing): n=985, text prior 48.7, 7B 44.8, best 65.4

> *"What direction is the dog moving towards?"* · *"Which direction does he move the spoon to empty it into
> the bowl?"* · *"Is the skier moving uphill or downhill?"*

**Understanding needed:** which way one entity's position changed. The options mix axes in one question:
left, right, toward the camera, not moving, absent.

**Sub-tasks:** [exists] -> ground(subject) -> select(instance: "in the foreground", "in the blue t-shirt",
"the smaller yellow monkey") -> entity_track(frame) -> label per axis, with a static verdict.

**Inputs:** one track, and a frame. 884 questions name no frame. Dyn-Bench calibration found the first
camera's axes match the ground truth for left/right, and here the clips are short, so this is the default
to check first. For ego object moves, the frame is the wearer's, i.e. the camera's.

**Outputs:** lateral, depth and vertical displacement each against a noise floor, plus a static verdict.
VLM4D needs the static verdict more than Dyn-Bench does: "not moving" is the gold in 18% of the
questions that offer it.

**Where AgentV5 stands:** covered by `entity_track` (scene / camera / image / own) and the
`subject_direction` playbook. The VLM4D-specific risks are the existence step (0a) and synthetic clips.
Synthetic clips are generated video: depth and pose estimators have not been tested on them.

## 2. `repetition_count` (**new**): n=145, text prior 42.8, 7B 38.6, best 51.0

> *"How many times does the horse jump to the left?"* · *"How many full revolutions does the dancer spin
> clockwise?"* · *"How many spoons of salt did he put in with his left hand?"*

**Understanding needed:** how many times an event happens, often filtered by direction or by hand. The
answer is 0 surprisingly often: 77 of the 256 counting questions.

**Sub-tasks:** ground(actor or object) -> event_series(actor, predicate) -> filter(direction / hand) ->
count(onsets).

**Inputs:** a per-frame signal whose peaks are the events: vertical position for jumps, accumulated yaw for
spins (revolutions = total yaw / 360), lateral displacement steps for "hit to the left", and hand-object
contacts for scoops and pick-ups (ego).

**Outputs:** the event times, each with its direction, and the count after the filter. The times matter so
Qwen can check them against the frames.

**Where AgentV5 stands:** `event_onset` returns only the **first** onset of a named predicate. Nothing
returns a series. Jumps and spins are geometric (track and orientation); scoops and pick-ups in ego video are
not, and stay with Qwen.

## 3. `count_with_predicate` (**new**): n=112, text prior 30.4, 7B 36.6, best 52.7

> *"How many people are moving to the left?"* · *"How many bears are facing away from the camera?"* ·
> *"How many flamingos are standing still?"*

**Understanding needed:** count the instances of a category that satisfy a per-instance condition. All 112
are exo; most have 4 small numeric options.

**Sub-tasks:** ground(category, all instances) -> per instance: entity_track or orientation ->
predicate(moving left / facing away / static) -> count.

**Outputs:** a table of instances, each with its measured direction or facing and a yes/no on the
predicate, plus the count and how stable it is.

**Where AgentV5 stands:** `count` counts distinct instances and `tracks_of` gives one track per instance.
The missing piece is composing them: a tool, or a `count` argument, that applies the direction or facing
measurement to each instance. This is the weakest kind for the text prior (30.4), so measurement has the
most room here.

## 4. `rotation_sense` (**new**): n=108, text prior 53.7, 7B 50.9, best 60.2

> *"Is the dancer spinning clockwise or counter-clockwise?"* · *"Are the horses moving clockwise or
> counter-clockwise around the track?"* · *"Which rotational direction is the person turning the knob?"*

**Understanding needed:** the sense of a rotation. There are three physically different cases:
- **Spin about the vertical** (dancers, cars, jet-ski): the sign of the yaw rate, presumably seen from above.
- **Circulation along a path** (horses around a track, a car on a roundabout): the sign of the area the
  top-down path encloses.
- **Rotation in the image plane** (knob, lid, stirring, steering wheel; mostly ego): the sign of the
  optical-flow curl inside the mask, as seen by the camera.

**Sub-tasks:** ground(subject or part) -> decide the case -> measure the sign -> label (cw / ccw / none).

**Where AgentV5 stands:** `orientation` reports heading turns in degrees about the vertical, which covers
the first case on clips where Orient-Anything resolves a front. A dancer's back and front are hard. The
`scene` frame of `entity_track` gives the path for the second case. Nothing measures image-plane rotation.
"Not spinning" is a frequent gold, so the static verdict needs a calibrated floor.

## 5. `traj_subject_own_frame` (existing): n=88, text prior 53.4, 7B 52.3, best 64.8

> *"Is the car driving forwards or backwards from its own perspective?"* · *"Is the cat turning to the left or
> right from its own perspective?"*

The paper's "perspective-aware" questions. Forward/backward needs the subject's facing
(`entity_track(frame=own)`, which uses Orient-Anything). Own-frame turning needs the heading *change*,
which `orientation` reports as `turn`. Covered. The answer sets add "moving straight and not turning" and
"not moving", so both static verdicts must be reported.

## 6. `camera_path` / `camera_translation` (existing; ego wearer): n=86 + 19, text prior 48.8 / 47.4, 7B 45.3 / 26.3

> *"Which way does the person turn to get the pan?"* · *"Which rotational direction does the person turn to go
> to the fridge?"* · *"How many times does he pan his head?"* (counting)

**Understanding needed:** the wearer's yaw (turns, head pans) or translation (walks toward the sink).
"Clockwise" for the wearer most plausibly means seen from above, i.e. a right turn. This is unverified;
check the sign against `camera_motion` pan on a handful of clips before trusting it.

**Where AgentV5 stands:** `camera_motion` measures exactly this. What is missing is 0b, getting there. Ego
clips run to 15 s with fast head motion, the worst case for pose estimation, so the reliability notes
matter.

## 7. `cam_subject_range` (existing): n=80, text prior 48.8, 7B 40.0, best 80.0

> *"Is the dog coming closer or moving further away?"* · *"Does the person move closer to the oven?"* (ego:
> the wearer and the oven)

Covered by `distance`. The large gap between the best models (75-80) and the 7B (40) suggests it is
solvable from appearance alone, i.e. apparent size. A measured range should push Qwen3-VL-8B toward
the top.

## 8. `relative_position` (**new**): n=55, text prior 56.4, 7B 47.3, best 67.3

> *"Where is the butter pack placed with respect to the bread slices?"* · *"On which side of the van does the
> cyclist go past?"* · *"Which side of the stove did the person place the oil jug?"*

**Understanding needed:** which side of B entity A is on (left / right / in front / behind / above /
below), at a moment (often the end, after placing) or during a pass. All 55 are ego, so "left of B"
is in the wearer's (camera's) view.

**Sub-tasks:** ground(A) -> ground(B) -> pick the moment (placement = end, pass = closest approach) ->
relative_side(A, B) in camera axes, with depth for front/behind.

**Where AgentV5 stands:** `distance` and `region_relation` give range, not side. Two camera-frame tracks
contain the answer, but no tool states it. A small `relative_side` tool, or a `side` line in `distance`'s
output, would cover this.

## 9. `subject_turn` (**new**): n=41, text prior 53.7, 7B 22.0, best 56.1

> *"Which way is the tractor turning towards?"* · *"Is the cat turning to the left or right from the camera
> perspective?"*

**Understanding needed:** whether the heading turns left, right or not at all, when no own frame is named.
The 7B is at **22%**, below random. The text prior (53.7, from "moving straight and not turning") beats
every model but one. Something here systematically fools a VLM, plausibly the image-left/right of the
motion being read as the turn.

**Sub-tasks:** ground(subject) -> entity_track(scene) -> velocity direction at the start vs the end ->
signed heading change against a floor -> label. "From the camera perspective" probably still means the
subject's turn as seen; unverified.

**Where AgentV5 stands:** `orientation` gives `turn` when a front resolves. The velocity-heading version
from the track (no orientation model needed) does not exist. It would serve both this kind and the
circulation case of `rotation_sense`.

## 10. `hand_attribution` (**new**): n=29, text prior 27.6, 7B 55.2, best 69.0

> *"Which hand does he use to drink from the cup?"* · *"What is the left hand doing?"*

Ego, apart from 3 questions. The wearer's left hand appears on the image's left, so SAM3 "hand" with the
image side, or "left hand" / "right hand" directly, could label the hands for Qwen. It is cheap to try, but
29 questions do not justify much. Until then it routes like `semantic` (vlm_only).

## 11. The small ones

- `landmark_relation` (25, ego): direction of a moved object relative to a named reference ("with respect
  to the sink"). Covered by `region_relation`, `distance` and `entity_track`. The text prior is very low
  (16.0): the options are hard to guess, so the measurement carries the answer.
- `semantic` (15): yes/no outcomes ("does the ball hit the backboard?"). vlm_only.
- `camera_viewpoint` (7): facing at the start or end, and "which side of the kayaker is the video taken
  from". Covered by `orientation`.
- `event_order` (6), `pair_relation` (4), `orbit_direction` (2): covered.
- `other` (9, ego cyclist clips): "what object is moving to the left?" and "is the runner faster than the
  cyclist?". They need every object's motion, or relative speed. Not worth a kind at this size; vlm_only.

## Build order this suggests

By questions reached per unit of work, and using only what the paper's models show:

1. **0a existence + 0b ego actor.** They touch 795 and 113 questions and are mostly prompt and
   evidence plumbing. 0a has to be calibrated against its 12% base rate.
2. **`count_with_predicate`** (112): a composition of existing tools. Lowest text prior, so a measurement
   has the most room.
3. **Velocity-heading change** (`subject_turn` 41, plus circulation in `rotation_sense`): one track
   computation; the 7B is far below chance on `subject_turn`.
4. **`repetition_count`** (145, the largest new kind): needs a real event-series detector. The geometric half
   (jumps, spins, steps) first, ego manipulation later.
5. **`relative_position`** (55): a small `side` output.
6. **Image-plane rotation** (part of `rotation_sense`) and **hands** (29): last.

Compare every result with the text prior (47.2% overall, 61.1% synthetic), not with 25%.
