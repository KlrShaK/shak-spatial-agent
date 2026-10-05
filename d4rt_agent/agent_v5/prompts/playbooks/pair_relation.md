- Call `distance` with a = the first entity and b = the second (for 'the two X': a = g1[0], b = g1[1]).
- Read the per-span readings in time order; a reversal (closer then farther) is not the same as 'closer'.
- If an entity is not found or the readings jump, `show` 'entities' to check the segmentation.
- A reversal is what options call alternating / fluctuating. For 'the two X' with only one instance found,
  `ground` X again at a time (at_s) where both are clearly visible, then measure.
