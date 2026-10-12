Evidence review -- one opportunity to repair the plan before answering.

Current kind: `{kind}`.
Measurement turns remaining after this review: {remaining}. Kind change: {reroute}.
Current evidence:
{evidence}

Only the current evidence listed above is valid; measurements from older entity bindings are superseded.

Check whether the evidence and frames distinguish the candidate claims. Check the intended entities,
coordinate frame, time span, and whether the question asks for net change, a sequence, or dominant motion.
Scalar distance cannot establish left/right position. Unavailable evidence cannot establish absence.

Choose `answer` if the usable evidence and frames suffice, or further tools cannot resolve the uncertainty.
Choose `measure` if one more measurement or inspection from the current kind's menu would resolve it.
Choose `reroute` only if another kind is needed; you will choose a family, then a kind from its small menu.
If no measurement turns remain, choose `answer` and retain any unresolved limitation.
Visual-only questions can be answered directly. Do not request geometry just because no tool was called.

{schemas}
Reply with brief reasoning and ONE JSON action. A repair must name the missing evidence in `missing`.
