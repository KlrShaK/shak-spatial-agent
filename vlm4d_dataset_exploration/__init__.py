"""VLM4D dataset exploration -- what does each question ask, and where would AgentV5 send it?

The third of these packages, after ``dsi_dataset_exploration`` (DSI-Bench) and ``dyn_dataset_exploration``
(Dyn-Bench). VLM4D (arXiv 2508.02095, ICCV 2025) has 1816 four-option questions on 1000 short clips:
DAVIS and YouTube-VOS (exo-centric), Ego4D (ego-centric) and Cosmos-generated synthetic video.

Three modules, in the order they run, none of them touching the network or a GPU:

1. ``taxonomy`` -- the deterministic rules: the paper's annotation type, the AgentV5 family/kind, the
   frame of reference, the time window, and each question's ordered sub-task program.
2. ``vlm4d_data`` -- loads the questions through AgentV5's loader (same ids), labels them, and measures
   the videos (``videos`` runs as a CPU job).
3. ``taxonomy_report`` -- the generated ``logs/TAXONOMY_REPORT.md``, including how the paper's 22 graded
   models fare per kind.

Hand-written analysis lives next to the generated files: ``logs/VLM4D_DATASET.md`` (the benchmark and its
conventions) and ``logs/BUCKET_REQUIREMENTS_V5.md`` (per kind: sub-tasks, inputs, outputs, AgentV5 gaps).
"""
