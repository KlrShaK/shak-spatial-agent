# Step 2 of decomposing a video question: fill the slots of its kind

The question is of kind `{kind}`: {kind_description}

The measurement for this kind is already fixed. What it still needs is read from the
question, and nothing else:

{slots}

Rules for every entity phrase:
- at most 5 words, a plain noun phrase an open-vocabulary segmenter can find in the frames
  ("person in dark jacket", "gray car", "dog");
- name the thing, not what it does ("person", not "person who walks away");
- copy the question's own words where they are enough; add one distinguishing detail only
  if several similar things are visible in the frames.

Reply with one or two sentences of reasoning, then exactly one JSON object and nothing after it:

{"slots": {<the slots above, as JSON>}}
