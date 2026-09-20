"""Agent v4: dynamic prompt composition over the SAM3 + OAV2 + D4RT pipeline.

v2 handed the model 32 frames, four free-form tools and a demand that it invent
its own measurement plan.  Most of what went wrong was planning, not measuring --
agents walked frames one at a time, re-queried targets that were never visible,
and ran out of budget before answering.

v4 inverts that.  The host decides what to measure, and does it:

    1. the agent classifies the question into one of four measurement buckets
    2. the host runs only the pipeline stages that bucket needs
    3. the agent names the subject, if the bucket has one
    4. the host hands back that one bucket's report, and the contract for it
    5. the agent answers

The agent still reasons in plain text before every action.  What changed is that
it receives the scene in stages, each one relevant to the step in front of it,
instead of all at once at the start.
"""
