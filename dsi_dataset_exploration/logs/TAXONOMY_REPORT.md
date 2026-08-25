# DSI-Bench task-type taxonomy

Exhaustive template scan: 1769 rows -> 34 distinct question templates (no LLM involved -- plain regex over every row). Blind LLM classification: 200 of 200 sampled questions classified.

## Taxonomy

- **subject_own_frame** — Trajectory of the subject relative to its own frame
  - cate 0 (Obj:static cam): 33/33 blind LLM agreement (100%) — confirmed present
  - cate 1 (Obj:moving cam): 34/34 blind LLM agreement (100%) — confirmed present
- **subject_observer_frame** — Trajectory of the subject relative to the observer/camera
  - status: absent from DSI-Bench as a standalone question type (no `cate` maps here by hypothesis; whether it appears as an implicit sub-quantity inside another category's distractor options is open -- see "Open questions")
- **distance_change** — Change of distance between the subject and the observer
  - cate 4 (Obj-Cam distance): 33/33 blind LLM agreement (100%) — confirmed present
- **camera_own_frame** — Trajectory of the camera/observer relative to its own frame
  - cate 2 (Cam:static scene): 33/33 blind LLM agreement (100%) — confirmed present
  - cate 3 (Cam:dynamic scene): 34/34 blind LLM agreement (100%) — confirmed present
- **camera_subject_frame** — Trajectory of the camera/observer relative to the subject
  - cate 5 (Obj-Cam orientation): 33/33 blind LLM agreement (100%) — confirmed present
- **observer_orient_rel_subject** — Orientation change of the observer relative to the subject (OpenCV coordinate convention)
  - status: absent from DSI-Bench as a standalone question type (no `cate` maps here by hypothesis; whether it appears as an implicit sub-quantity inside another category's distractor options is open -- see "Open questions")
- **subject_orient_over_time** — Change of orientation of the subject through time
  - status: absent from DSI-Bench as a standalone question type (no `cate` maps here by hypothesis; whether it appears as an implicit sub-quantity inside another category's distractor options is open -- see "Open questions")
- **observer_orient_over_time** — Change of orientation of the observer through time
  - status: absent from DSI-Bench as a standalone question type (no `cate` maps here by hypothesis; whether it appears as an implicit sub-quantity inside another category's distractor options is open -- see "Open questions")
- **subject_orient_rel_observer** — Orientation change of the subject relative to the orientation of the observer
  - status: absent from DSI-Bench as a standalone question type (no `cate` maps here by hypothesis; whether it appears as an implicit sub-quantity inside another category's distractor options is open -- see "Open questions")

## cate ↔ bucket mapping

| cate | category | hypothesis bucket | n | agree | OTHER | agreement % |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Obj:static cam | subject_own_frame | 33 | 33 | 0 | 100% |
| 1 | Obj:moving cam | subject_own_frame | 34 | 34 | 0 | 100% |
| 2 | Cam:static scene | camera_own_frame | 33 | 33 | 0 | 100% |
| 3 | Cam:dynamic scene | camera_own_frame | 34 | 34 | 0 | 100% |
| 4 | Obj-Cam distance | distance_change | 33 | 33 | 0 | 100% |
| 5 | Obj-Cam orientation | camera_subject_frame | 33 | 33 | 0 | 100% |

## Bucket frequency across the sample

| bucket | description | times chosen |
| --- | --- | --- |
| subject_own_frame | Trajectory of the subject relative to its own frame | 67 |
| subject_observer_frame | Trajectory of the subject relative to the observer/camera | 0 |
| distance_change | Change of distance between the subject and the observer | 33 |
| camera_own_frame | Trajectory of the camera/observer relative to its own frame | 67 |
| camera_subject_frame | Trajectory of the camera/observer relative to the subject | 33 |
| observer_orient_rel_subject | Orientation change of the observer relative to the subject (OpenCV coordinate convention) | 0 |
| subject_orient_over_time | Change of orientation of the subject through time | 0 |
| observer_orient_over_time | Change of orientation of the observer through time | 0 |
| subject_orient_rel_observer | Orientation change of the subject relative to the orientation of the observer | 0 |
| OTHER | none of the above fit | 0 |

## Open questions

- cate 5 (`Obj-Cam orientation`) is hypothesized as `camera_subject_frame` (observer position expressed in the subject's facing-defined frame), not `observer_orient_rel_subject` -- see its agreement row above and the review queue for any cate-5 disagreements.
- Buckets `subject_observer_frame`, `observer_orient_rel_subject`, `subject_orient_over_time`, `observer_orient_over_time`, and `subject_orient_rel_observer` have no standalone DSI-Bench question template. Whether any of them are tested implicitly (e.g. the rotating/orbiting distractor options in cate 0-3 may require knowing orientation-through-time to rule out) is not resolved by this pass and needs a dedicated look at the raw option text.
- `d4rt_agent/prompts/dsi_bench_system.md`'s existing "Question -> measurement" table has no row for cate 5's side-to-side transition phrasing -- a gap this taxonomy confirms independently of the LLM pass.

## Review queue

0 disagreements logged, 0 still awaiting `human_verdict`.

## Future work (explicitly out of scope for this pass)

Wiring this taxonomy into a conditional/task-adaptive system prompt for the D4RT agent is not designed or implemented here -- this report is the identification step only.
