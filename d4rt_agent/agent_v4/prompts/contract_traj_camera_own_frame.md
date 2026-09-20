### How to read this measurement

This is the observer's own motion, in the frame it started in. `forward`,
`left` and `right` are the camera's own heading at frame 0 — not the subject's,
and not the screen's.

**`up` and `down` are measured against gravity**, not against the camera's tilt,
whenever the report says so. Those are the same thing only for a level camera. A
camera pitched down 50° that descends straight down would otherwise split that
descent almost equally between its own "down" and its own "forward", and read as
a tie between two different answers instead of a plain descent. If the report
instead warns that the world vertical could not be recovered, then up/down are
the camera's own axes and a tilted camera may show vertical motion partly as
forward/backward — say so if you rely on it.

The path is cut into 10 equal-duration segments. Each segment gives a direction,
a distance, and how far the camera turned during it. The prose merges runs of
segments that share a direction; the per-segment table underneath is the same
data unmerged.

**Angles.** All three are changes from frame 0, so they start at zero. Where the
frame was levelled, `pan` is a true compass-style turn about the vertical and
`tilt` a true angle above or below the horizon.

- `pan` — turning left or right. **Positive means the camera turned to its own
  right.**
- `tilt` — positive is upward.
- `roll` — positive means the camera's right side dipped.

**Translation and rotation together are what identify the motion.** Read them as
a pair, because several options differ only in their combination:

- Moving sideways with **no significant pan** is a straight sideways move
  (a dolly or a truck). The camera keeps facing the same way.
- Moving sideways **while panning in the direction that keeps the scene in
  view** is an *orbit*: the camera is circling something rather than sliding
  past it. Moving left while panning right, or moving right while panning left,
  is the signature — the pan compensates for the translation.
- Panning with **no significant translation** is the camera turning on the spot,
  not moving.
- Neither above its noise floor means the observer did not move.

**Which way an orbit turns**, seen from above, follows from the pair — the pan is
what keeps the subject in view as the camera slides around it:

- translating **left** while panning **right** (positive pan) → **clockwise**
- translating **right** while panning **left** (negative pan) → **counter-clockwise**

Read the sign off the reported numbers, not off the frames. Do not invert it
because a direction "looks" opposite on screen: the scene sweeping one way across
the image is what the camera doing the other thing looks like, and that is the
mistake this rule exists to prevent.

**Noise floors.** A `noise floor` in units is given for translation, and one in
degrees for each angle. Anything at or below its floor did not happen. If the
report opens by saying the camera is STATIONARY, the per-segment directions are
jitter and you must not read a direction out of them.
