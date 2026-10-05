"""The 4D backbone behind every geometric tool, chosen per run: ``--backbone point4d|d4rt``
(``BACKBONE=`` in the slurm launchers).

Both adapters return the same arrays (the ``scene.npz`` layout of the research pipeline) and a
metric scale with its source named, so nothing above this package knows which one ran. What does
differ, and is recorded in every scene: how many frames it covers (Point4D up to 400 GF frames,
D4RT at most 32) and where its metres come from (Point4D: DA3-Nested depth ratio; D4RT: its own
depth units, taken as metres).
"""

from __future__ import annotations

from d4rt_agent.agent_v5.backbones.base import BACKBONES, Backbone, SceneCache, get_backbone

__all__ = ["BACKBONES", "Backbone", "SceneCache", "get_backbone"]
