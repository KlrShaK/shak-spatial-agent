"""Dyn-Bench dataset exploration -- what does each question actually measure?

The sibling package ``dsi_dataset_exploration`` asked that question of DSI-Bench
and got four confirmed measurement buckets, which are what
``d4rt_agent/agent_v4/buckets.py`` implements today. This package asks it of
Dyn-Bench (arXiv 2603.12746, "Thinking in Dynamics"), whose 7000 VQA questions
are broader in kind, not just in count.

Three modules, in the order they run:

1. ``dyn_bench_data`` -- loads the parquet, resolves every question to its media
   across eight differently-named source datasets, runs the deterministic scans
   (statistics, question families, template collapse), and draws the
   family-stratified 600-question sample. No network.
2. ``taxonomy_classify`` -- sends each sampled question, text and options only,
   to a small LLM that labels it on two axes at once: the paper's own nine
   sub-tasks, and our measurement-centric buckets. Network, ~600 calls.
3. ``taxonomy_report`` -- renders the cross-tab and the AgentV4 coverage
   verdict. No network.

Scope: the 7000-row VQA track only. The 3000 mask/grounding rows are named in
the report as the largest untouched capability gap, not analysed here.
"""
