# Step 1 of decomposing a video question: which kind of question is it?

You are not answering the question. You are deciding what would have to be measured to
answer it, so that a measurement pipeline can do the measuring.

Read the question stem. Name the quantity it asks about, whose quantity it is (the camera,
one thing in the scene, two things, a category of things, or nothing measurable at all),
and relative to what. Ignore the surface wording of the options: they show what an answer
looks like, not what is measured.

Choose exactly one kind:

{kinds}

If none fits, choose `other`.

Reply with one or two sentences of reasoning, then exactly one JSON object and nothing after it:

{"kind": "<one kind name, spelled exactly as listed>", "justification": "<one line>"}
