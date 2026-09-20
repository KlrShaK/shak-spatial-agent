### How to read this measurement

This is the subject's own motion, expressed in the subject's own starting pose:
`forward` means the direction **the subject was facing**, not the direction the
camera points. Camera motion has already been cancelled, so this describes what
the subject did regardless of what the observer was doing.

The path is cut into 10 equal-duration segments, each with a direction, a
distance, and how far the subject's own facing turned during it. The prose
merges runs of segments sharing a direction; the table underneath is unmerged.

**Heading.** Positive means the subject turned **to its own right**. It is
measured relative to the subject's facing at the frame named as the pose frame.

**Translation and turning together identify the motion.** Options frequently
combine them:

- Moving with **no significant heading change** is straight-line motion.
- Moving forward **while the heading changes steadily** is a curved path —
  "moving forward and turning left/right".
- A large heading change with **no significant translation** is the subject
  turning on the spot rather than going anywhere.
- Neither above its floor means the subject stayed put.

**Watch the heading's span.** Orientation is only available on frames where the
estimator resolved a single unambiguous front. If the report says the heading
was measured over only part of the clip, it describes that part only — do not
report it as a whole-clip turn, and say so if you rely on it.

**Noise floors.** Translation has a floor in units; heading has one in degrees.
At or below the floor is not a change. If the report opens by calling the
subject STATIONARY, the per-segment directions are jitter.
