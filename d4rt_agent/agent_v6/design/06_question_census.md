# 06 — What the benchmarks ask, by physical quantity

Context for planning (tracker PC, 2026-10-17). Every question in the three benchmarks (10,585) is
grouped by the physical quantity that decides its answer. Each group gets the V6 tool that measures
it and the evidence on whether measuring has helped so far. Produced by
[analysis/quantity_census.py](analysis/quantity_census.py) from existing labels, with no new model
calls:

- DSI's category for all 1,769 questions;
- the Dyn decomposition census for all 7,000 (rule labels where rules apply, a Qwen text-only label
  otherwise);
- the VLM4D exploration manifest for all 1,816, including its six kinds that V5 lacked.

## The table

Accuracy columns, all in percent, with chance at 25%:

- **frames, letters, with measurement:** the D7 study's 2,800-question Dyn eval, for frames-only
  Qwen, Qwen's option log-probabilities, and Qwen answering with our measurement in the prompt.
  "With measurement" counts only questions that had a measurement; that count is in brackets.
- **V5 Dyn:** AgentV5 legacy (Point4D) on the same 2,800 questions.
- **V5 VLM4D:** AgentV5 legacy on all 1,816 VLM4D questions.

| Quantity | V6 tool | DSI | Dyn | VLM4D | All | frames | letters | with measurement | V5 Dyn | V5 VLM4D |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Two entities, or an entity and a landmark: distance, relative position | `track_entity` (others) | 0 | 1,740 | 84 | 1,824 | 61 | 63 | 58 (661) | 57 | 49 |
| An entity's motion and direction | `track_entity` | 767 | 398 | 1,073 | 2,238 | 56 | 56 | 54 (132) | 51 | 59 |
| Camera motion: path, translation, turning, orbit | `track_camera` | 779 | 836 | 107 | 1,722 | 45 | 49 | **57** (319) | 56 | 67 |
| Entity and camera: distance, height, side filmed | `track_entity` (camera / subject) | 223 | 1,002 | 87 | 1,312 | 48 | 52 | 53 (351) | 48 | 77 |
| Events, actions, repetitions, hands, appearance | ask-video tool / VLM | 0 | 669 | 195 | 864 | **79** | **79** | 73 (97) | 71 | 44 |
| An entity's speed profile, incl. metric speed | `track_entity` (speed section) | 0 | 681 | 0 | 681 | **60** | 59 | 56 (266) | 56 | — |
| On-screen size and visibility | `get_segmentation_mask` | 0 | 563 | 0 | 563 | 72 | 72 | 71 (222) | 70 | — |
| Camera tempo and steadiness | `track_camera` | 0 | 547 | 0 | 547 | 43 | 44 | **54** (210) | 50 | — |
| Counting instances, incl. with a motion predicate | inventory + `track_multiple_entities`; VLM counts | 0 | 223 | 112 | 335 | 55 | **63** | 44 (32) | 48 | 39 |
| Lens zoom | `track_camera_zoom` | 0 | 170 | 0 | 170 | 77 | 77 | **89** (74) | 86 | — |
| Rotation sense, turning, facing | `track_orientation` | 0 | 0 | 149 | 149 | — | — | — | — | 35 |
| End view returns to the start view | `check_scene_revisit` | 0 | 127 | 0 | 127 | 50 | 61 | **71** (56) | 66 | — |
| No label | — | 0 | 44 | 9 | 53 | | | | | |

DSI's measurement evidence is in [01_baseline.md](01_baseline.md): V4 against the tool-free control,
averaged over the four renderings. Camera motion 80/71 vs 38/38 (c2/c3); entity–camera 75/53 vs
61/25 (c4/c5); entity motion 52/53 vs 50/40 (c0/c1).

## What it says

**Four quantities cover two thirds of all questions:** an entity's motion, relations between two
things, camera motion, and entity–camera relations (7,096 of 10,585). They map onto just two
tools, `track_entity` and `track_camera`. These tools and their reports matter most.

**Measuring clearly helps for the camera:** motion (+8 to +12 over frames and letters), tempo
(+10), zoom (+12), revisit (+10 to +21). It also helps on DSI's camera and entity–camera
categories, where the reversed renderings remove the prior.

**Measuring was neutral or worse where V5's tool was the wrong observable.** Relations between two
things scored below frames because a scalar distance cannot tell left from right; `track_entity`'s
signed relative geometry is the fix. Entity motion on Dyn is level with frames, while on DSI it
holds up under reversal.

**The VLM should decide two groups:**

- **counting:** letters 63 vs 44 with measurement;
- **events and actions:** frames 79 vs 73 with measurement.

This supports #13 (the VLM counts) and the new ask-video tool (#44). Speed is slightly worse with
measurement (56 vs 60); its estimator is the issue ([Assumptions.md](Assumptions.md) A1).

**Rotation is the thinnest group:** 149 questions, all VLM4D, where V5 scored 35%. Its tool,
`track_orientation`, depends on Orient-Anything finding a front.

## Caveats

- Labels come from rules and a text-only Qwen pass, not human annotation. The VLM4D rules were
  hand-audited at 95% (124/131).
- The Dyn accuracy columns come from the 2,800-question eval split, not all 7,000.
- "With measurement" uses the D7 study's measurements, not V6's tools. It shows where measuring
  has helped so far, not what V6 will score.
