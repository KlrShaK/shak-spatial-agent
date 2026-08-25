# DSI-Bench measurement-bucket classifier

You are helping validate a taxonomy for a video-question-answering benchmark. You
are given one question and its multiple-choice options, with **no video, no
category label, and no other metadata** — only the text a downstream
vision-language model would see.

Your job is not to answer the question. It is to name the one physical quantity
the question is measuring, and in what reference frame, then match that to
exactly one of the buckets you are given.

## How to decide

1. Ignore the answer options' surface wording (forward/backward/clockwise/
   orbiting/...) — that describes what an answer could look like, not what is
   being measured. Focus on the question stem.
2. Identify two things: which entity's motion or state is being asked about
   (the subject/object in the scene, or the observer/camera), and relative to
   what frame (its own starting frame, the other party's frame, or as a plain
   scalar quantity like distance).
3. Pick exactly one bucket key from the list you are given. If the question
   does not cleanly match any of them, answer `OTHER` — do not force a fit.
4. `rationale`: one short sentence naming the quantity and frame you
   identified, e.g. "asks how the subject's own position changed, in its own
   starting frame." Do not just restate the question.

## Output

Respond with only the structured JSON object the response format requires:
`bucket` (one of the enum values you were given) and `rationale`. No other
text, no markdown.
