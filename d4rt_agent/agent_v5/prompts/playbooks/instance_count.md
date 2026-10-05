- Call `count`: distinct instances with their visible spans.
- The segmenter can miss small or occluded instances and split one instance in two; `show` 'entities' and check the
  frames before trusting an unusual count.
- If you re-ground, use the plain category noun from the question ('cow', 'person'), never a description:
  a longer phrase finds fewer instances and undercounts.
