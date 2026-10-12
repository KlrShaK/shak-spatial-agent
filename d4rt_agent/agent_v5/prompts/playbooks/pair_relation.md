- For closer/farther or separation, call `distance` with a = the first entity and b = the second
  (for 'the two X': a = g1[0], b = g1[1]).
- For left/right, above/below or relative position, call `relative_position` with those same operands.
  Choose `image` for position in the picture, `scene` for fixed initial-camera axes with camera motion
  removed, or `camera` for the current camera axes. State your frame interpretation in the justification.
  Depth along camera axes is not ahead/behind relative to another entity's facing; that body-relative
  distinction is unsupported here and must be judged from the frames. Distance alone cannot decide it.
- Read the per-span readings in time order; a reversal (closer then farther) is not the same as 'closer'.
- If an entity is not found or the readings jump, `show` 'entities' to check the segmentation.
- A reversal is what options call alternating / fluctuating. For 'the two X' with only one instance found,
  `ground` X again at a time (at_s) where both are clearly visible, then measure.
- A selected pair cannot establish the arrangement of an entire group; use the frames for group claims.
