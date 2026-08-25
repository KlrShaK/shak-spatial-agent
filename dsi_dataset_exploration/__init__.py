"""DSI-Bench task-type taxonomy exploration.

Self-contained research pipeline: identify which of the nine tool/measurement-
centric task buckets DSI-Bench actually tests, via an exhaustive question-
template scan plus a blind LLM classification pass over a category-stratified
sample. Kept separate from ``d4rt_agent/dsi_bench_*`` (the production eval
pipeline) since this is a one-off taxonomy exercise, not part of the agent's
eval harness.

See ``taxonomy_data.py``, ``taxonomy_classify.py``, and ``taxonomy_report.py``.
"""
