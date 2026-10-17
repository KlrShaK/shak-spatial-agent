# 14 — From measurement to answer

Theme 5 of the discussion. It covers tracker point P2: the measurement reasoning matches the
reference, then the final step flips to a wrong answer. Status: **settled** on 2026-10-17 (decision
log #64–68).

## Why answers flipped

- **A prompt rule invited it.** V5's system prompt said: *"If a tool's result clearly contradicts
  what the frames show, rely on the frames."* V4's said the opposite: *"You cannot measure 3D motion
  by looking … an answer that reads plausible from the frames alone is indistinguishable from a
  guess."* For 3D motion, an impression from the frames is mostly a prior. The tool-free control
  falls from 58.5% to 25.0% on time-reversed DSI video; V4 falls only to 54.5%.
- **V5 legacy re-sent the frames with every report at the answer turn**, giving that prior a fresh
  chance to override the measurement right at the end.
- **Compositional V5 failed the other way.** Its fresh text-only packet had no images on 136 of 252
  DSI questions. It scored 67% when every measurement worked and 32–39% when anything failed.
- **Five workflow-pilot regressions** had a review that endorsed the correct answer and a final
  answer that contradicted it.

## The decisions

**Same conversation through the answer (#64).** No fresh packet, no re-sent dossier. The answer turn
sees what the agent already has: system prompt, caption (#36), question and options, the 8
keyframes (#32), its plan, every report, and any frames it requested.

**No frames added at the answer (#65).** Only frames the agent asked for during its turns. Your
empirical observation: performance drops as visual input grows, and the answer can often be inferred
from the measurements. The data agrees. The all-frames official protocol scored no better than 16
frames on Dyn (60.6% vs 59.7%), and on questions where V5 made no tool call, its re-sent frames
scored 140/192 against frames-only Qwen's 148/192. The 8 keyframes remain in context, so a failed
measurement never leaves the answer without any image. The agent can still ask for `get_frames` or
`ask_video` first (#53).

**Precedence rule (#66).** Stated in the answer prompt and the system prompt:

- 3D motion, direction, distance and facing: **the measurement decides**, unless its report is
  unavailable or the agent has shown that it measured the wrong object (`get_segmentation_mask`);
- visual facts (events, actions, appearance): frames and `ask_video` decide;
- an unavailable measurement is not evidence of absence or of no motion.

This replaces V5's "rely on the frames if they contradict".

**Answer format (#67).** V4's form: the letter, the option's text, then what decided it. For
example: "C: Moving downward. Net 2.6 down (noise 0.14)." The JSON is `{"letter": ..., "because":
...}`. The template shows a placeholder, never a filled example letter (V5's `"letter": "A"`
coincided with 1,331 predicted A's against 1,052 true). Malformed answers go through #63.

**Context budget (#68).** 32k tokens for the whole conversation (V5 exceeded its 16k budget at
18–20k). When it fills, the oldest requested images are dropped first. Report text and the plan are
always kept.

## Implementation notes

- **Answer prompt.** Short: the question and options again, the precedence rule, the format, and
  "match what you measured to the options; pick the best match or eliminate". Reports never name
  options (#6), so the matching happens here.
- **Flip audit, offline.** The `because` field names what decided the answer. Evaluation counts
  answers whose stated reason contradicts the chosen option, and answers whose last report's
  headline contradicts the chosen option without an override reason. This makes P2 measurable.
- **Trimming order.** Requested images are dropped oldest first. Never dropped: the 8 keyframes, any
  report, the plan.
