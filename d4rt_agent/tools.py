"""Tool surface exposed to the VLM orchestrator.

Provides (1) OpenAI/Qwen-style function schemas and (2) a dispatcher that executes
a named tool call against a :class:`DemoGeometry` (predicted) backend and returns a
JSON-serializable result. All numbers are metric (meters, seconds, m/s).
"""

from __future__ import annotations

import json
from typing import Any

from .geometry_tools import DemoGeometry


# -- function schemas (Qwen3-VL / OpenAI tool-calling format) --------------
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "ground_object",
            "description": (
                "Snap an image pixel (u, v) at frame t to the nearest tracked 3D point. "
                "Call this first to convert an object you see into a track_id you can query. "
                "u is the horizontal pixel (0=left), v vertical (0=top)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "u": {"type": "number", "description": "horizontal pixel, 0..width"},
                    "v": {"type": "number", "description": "vertical pixel, 0..height"},
                    "t": {"type": "integer", "description": "frame index"},
                },
                "required": ["u", "v", "t"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "position",
            "description": "3D position (meters, x y z) of a track at frame t.",
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id": {"type": "integer"},
                    "t": {"type": "integer"},
                },
                "required": ["track_id", "t"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "distance",
            "description": "Straight-line 3D distance in meters between two tracks at frame t.",
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id_a": {"type": "integer"},
                    "track_id_b": {"type": "integer"},
                    "t": {"type": "integer"},
                },
                "required": ["track_id_a", "track_id_b", "t"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "displacement",
            "description": (
                "Straight-line distance in meters between a track's positions at t0 and t1. "
                "Use this for start-to-end separation, not total distance travelled."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id": {"type": "integer"},
                    "t0": {"type": "integer"},
                    "t1": {"type": "integer"},
                },
                "required": ["track_id", "t0", "t1"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "path_length",
            "description": (
                "Accumulated 3D distance travelled in meters over consecutive visible frames. "
                "Use this for distance covered or travelled, not start-to-end separation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id": {"type": "integer"},
                    "t0": {"type": "integer"},
                    "t1": {"type": "integer"},
                },
                "required": ["track_id", "t0", "t1"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "speed",
            "description": "Average speed in m/s of a track between frame t0 and t1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id": {"type": "integer"},
                    "t0": {"type": "integer"},
                    "t1": {"type": "integer"},
                    "mode": {"type": "string", "enum": ["average", "path"]},
                },
                "required": ["track_id", "t0", "t1"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "is_moving",
            "description": "Whether a track is a moving (dynamic) or static object.",
            "parameters": {
                "type": "object",
                "properties": {"track_id": {"type": "integer"}},
                "required": ["track_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trajectory_summary",
            "description": "Compact motion summary of a track over the whole clip.",
            "parameters": {
                "type": "object",
                "properties": {"track_id": {"type": "integer"}},
                "required": ["track_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Return the final answer to the user's question and stop.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    },
]


class ToolDispatcher:
    """Executes named tool calls against a predicted-geometry backend."""

    def __init__(self, geometry: DemoGeometry) -> None:
        if geometry.source != "pred":
            raise ValueError("ToolDispatcher must wrap the 'pred' backend")
        self.g = geometry
        self.call_log: list[dict] = []

    @property
    def meta(self) -> dict:
        return {
            "num_frames": self.g.num_frames,
            "fps": self.g.fps,
            "width": self.g.width,
            "height": self.g.height,
        }

    def dispatch(self, name: str, args: dict[str, Any]) -> dict:
        try:
            result = self._run(name, args)
            record = {"tool": name, "args": args, "result": result}
        except Exception as exc:  # surface tool errors back to the model
            record = {"tool": name, "args": args, "error": f"{type(exc).__name__}: {exc}"}
        self.call_log.append(record)
        return record

    def _run(self, name: str, args: dict[str, Any]) -> dict:
        g = self.g
        if name == "ground_object":
            hit = g.nearest_track(args["u"], args["v"], args["t"])
            return {"track_id": hit.track_id, "pixel_dist": round(hit.pixel_dist, 2),
                    "snapped_uv": [round(hit.uv[0], 1), round(hit.uv[1], 1)]}
        if name == "position":
            x, y, z = g.position(args["track_id"], args["t"])
            return {"xyz_m": [round(x, 4), round(y, 4), round(z, 4)]}
        if name == "distance":
            return {"distance_m": round(g.distance(args["track_id_a"], args["track_id_b"], args["t"]), 4)}
        if name == "displacement":
            result = g.endpoint_measurement(args["track_id"], args["t0"], args["t1"])
            # Preserve the original result key for existing callers.
            result["displacement_m"] = result["endpoint_displacement_m"]
            return result
        if name == "path_length":
            return g.path_measurement(args["track_id"], args["t0"], args["t1"])
        if name == "speed":
            return {"speed_mps": round(g.speed(args["track_id"], args["t0"], args["t1"], args.get("mode", "average")), 4)}
        if name == "is_moving":
            return g.motion_label(args["track_id"])
        if name == "trajectory_summary":
            return g.trajectory_summary(args["track_id"])
        raise ValueError(f"unknown tool: {name}")


def tools_json() -> str:
    return json.dumps(TOOL_SCHEMAS, indent=2)
