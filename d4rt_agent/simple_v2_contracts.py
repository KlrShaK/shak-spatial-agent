"""CPU-only contracts shared by the live D4RT/Qwen v2 agent.

This module deliberately has no torch or transformers imports.  Sampling, action
validation, grounding policies, and numerical execution can therefore be tested on
a login node without loading either model.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


NUM_SAMPLED_FRAMES = 32
POINT_MODES = ("centroid", "ensemble5")
ENSEMBLE_SEED = 42
ENSEMBLE_RADIUS_PX = 12.0


@dataclass(frozen=True)
class SampledVideo:
    """One immutable temporal sample shared by Qwen and D4RT."""

    video_path: Path
    frames_rgb: np.ndarray
    original_indices: tuple[int, ...]
    total_original_frames: int
    fps: float
    width: int
    height: int

    @property
    def sampled_indices(self) -> tuple[int, ...]:
        return tuple(range(NUM_SAMPLED_FRAMES))

    def mapping(self) -> list[dict[str, int]]:
        return [
            {"sampled_frame": sampled, "original_frame": original}
            for sampled, original in enumerate(self.original_indices)
        ]


def uniform_sample_indices(num_frames: int, count: int = NUM_SAMPLED_FRAMES) -> tuple[int, ...]:
    """Return ``round(linspace(0, N - 1, count))`` exactly."""

    num_frames = int(num_frames)
    count = int(count)
    if count <= 0:
        raise ValueError("sample count must be positive")
    if num_frames < count:
        raise ValueError(
            f"simple v2 requires at least {count} frames; video has {num_frames}"
        )
    indices = np.rint(np.linspace(0, num_frames - 1, count, dtype=np.float64)).astype(np.int64)
    if len(np.unique(indices)) != count:
        raise RuntimeError("uniform sampling unexpectedly produced duplicate frame indices")
    return tuple(int(value) for value in indices.tolist())


def sample_video_cpu(video_path: str | Path) -> SampledVideo:
    """Decode on CPU and select the one 32-frame clip used by both models."""

    import cv2

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) < NUM_SAMPLED_FRAMES:
        raise ValueError(
            f"simple v2 requires at least {NUM_SAMPLED_FRAMES} frames; "
            f"{path} decoded {len(frames)}"
        )
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise RuntimeError(f"video changes resolution between frames: {path}")
    indices = uniform_sample_indices(len(frames))
    sampled = np.stack([frames[index] for index in indices], axis=0).astype(np.uint8, copy=False)
    if not math.isfinite(fps) or fps <= 0:
        fps = 15.0
    return SampledVideo(
        video_path=path,
        frames_rgb=sampled,
        original_indices=indices,
        total_original_frames=len(frames),
        fps=fps,
        width=int(sampled.shape[2]),
        height=int(sampled.shape[1]),
    )


# These are the only actions described to Qwen.  In particular, query_d4rt has no
# point_mode field; the policy is fixed in the host process before Qwen is loaded.
ACTION_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "name": "inspect_frames",
        "description": "Inspect selected sampled video frames in more detail.",
        "parameters": {
            "type": "object",
            "required": ["frame_indices", "justification"],
            "properties": {
                "frame_indices": {"type": "array", "items": {"type": "integer"}},
                "justification": {"type": "string"},
            },
        },
    },
    {
        "name": "query_d4rt",
        "description": "Query live 3D positions for a box-grounded object point.",
        "parameters": {
            "type": "object",
            "required": [
                "label", "bbox_2d_1000", "t_src", "t_tgt", "t_cam", "justification"
            ],
            "properties": {
                "label": {"type": "string"},
                "bbox_2d_1000": {
                    "type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4
                },
                "t_src": {"type": "integer", "minimum": 0, "maximum": 31},
                "t_tgt": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 31},
                },
                "t_cam": {"type": "integer", "enum": [0]},
                "justification": {"type": "string"},
            },
        },
    },
    {
        "name": "python_math",
        "description": "Run restricted numerical calculations over prior tool outputs.",
        "parameters": {
            "type": "object",
            "required": ["bindings", "code", "justification"],
            "properties": {
                "bindings": {"type": "object"},
                "code": {"type": "string"},
                "justification": {"type": "string"},
            },
        },
    },
    {
        "name": "final_answer",
        "description": "Return one supported numerical answer and cite its evidence.",
        "parameters": {
            "type": "object",
            "required": ["value", "unit", "evidence_ids", "limitations"],
            "properties": {
                "value": {"type": "number"},
                "unit": {"type": "string"},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "string"},
            },
        },
    },
)


def _require_justification(arguments: Mapping[str, Any]) -> str:
    value = arguments.get("justification")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("every non-final action requires a short justification")
    if len(value) > 500:
        raise ValueError("justification is too long (maximum 500 characters)")
    return value.strip()


def validate_action(action: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Validate one Qwen action and normalize its argument types."""

    if not isinstance(action, Mapping):
        raise ValueError("action must be a JSON object")
    def _contains_host_policy(value: Any) -> bool:
        if isinstance(value, Mapping):
            return "point_mode" in value or any(_contains_host_policy(child) for child in value.values())
        if isinstance(value, (list, tuple)):
            return any(_contains_host_policy(child) for child in value)
        return False

    if _contains_host_policy(action):
        raise ValueError("Qwen must never select or emit point_mode")
    name = action.get("action", action.get("name"))
    arguments = action.get("arguments")
    if name not in {item["name"] for item in ACTION_SCHEMAS}:
        raise ValueError(f"unsupported action: {name!r}")
    if not isinstance(arguments, Mapping):
        raise ValueError("action.arguments must be an object")
    args = dict(arguments)

    if name == "inspect_frames":
        _require_justification(args)
        indices = args.get("frame_indices")
        if not isinstance(indices, list) or not indices or len(indices) > NUM_SAMPLED_FRAMES:
            raise ValueError("frame_indices must be a non-empty list of at most 32 frames")
        args["frame_indices"] = [int(value) for value in indices]
        if any(value < 0 or value >= NUM_SAMPLED_FRAMES for value in args["frame_indices"]):
            raise ValueError("inspect frame index is outside [0, 31]")
    elif name == "query_d4rt":
        _require_justification(args)
        label = args.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("query_d4rt.label must be non-empty")
        bbox = args.get("bbox_2d_1000")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("bbox_2d_1000 must be [x_min, y_min, x_max, y_max]")
        bbox = [float(value) for value in bbox]
        if not all(math.isfinite(value) and 0.0 <= value <= 1000.0 for value in bbox):
            raise ValueError("bbox_2d_1000 coordinates must be finite and inside [0, 1000]")
        if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            raise ValueError("bbox_2d_1000 min coordinates must not exceed max coordinates")
        args["bbox_2d_1000"] = bbox
        args["t_src"] = int(args.get("t_src"))
        targets = args.get("t_tgt")
        if not isinstance(targets, list) or not targets:
            raise ValueError("t_tgt must be a non-empty list")
        args["t_tgt"] = [int(value) for value in targets]
        if not (0 <= args["t_src"] < NUM_SAMPLED_FRAMES):
            raise ValueError("t_src is outside [0, 31]")
        if any(value < 0 or value >= NUM_SAMPLED_FRAMES for value in args["t_tgt"]):
            raise ValueError("t_tgt contains a frame outside [0, 31]")
        if len(set(args["t_tgt"])) != len(args["t_tgt"]):
            raise ValueError("t_tgt must not contain duplicates")
        args["t_cam"] = int(args.get("t_cam"))
        if args["t_cam"] != 0:
            raise ValueError("simple v2 fixes t_cam=0")
    elif name == "python_math":
        _require_justification(args)
        if not isinstance(args.get("bindings"), Mapping):
            raise ValueError("python_math.bindings must be an object")
        if not isinstance(args.get("code"), str) or not args["code"].strip():
            raise ValueError("python_math.code must be non-empty")
    else:
        value = float(args.get("value"))
        if not math.isfinite(value):
            raise ValueError("final answer value must be finite")
        unit = args.get("unit")
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError("final answer unit must be non-empty")
        evidence = args.get("evidence_ids")
        if not isinstance(evidence, list) or not evidence or not all(isinstance(x, str) for x in evidence):
            raise ValueError("final answer must cite evidence_ids")
        limitations = args.get("limitations")
        if not isinstance(limitations, str):
            raise ValueError("final answer limitations must be a string")
        args.update(value=value, unit=unit.strip(), evidence_ids=evidence, limitations=limitations.strip())
    return str(name), args


def bbox_1000_to_pixels(
    bbox_2d_1000: Sequence[float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert Qwen's [0,1000] box to original-video pixel coordinates."""

    x0, y0, x1, y1 = (float(value) for value in bbox_2d_1000)
    sx = float(max(int(width) - 1, 1)) / 1000.0
    sy = float(max(int(height) - 1, 1)) / 1000.0
    return x0 * sx, y0 * sy, x1 * sx, y1 * sy


def grounding_points(
    point_mode: str,
    bbox_2d_1000: Sequence[float],
    width: int,
    height: int,
    *,
    seed: int = ENSEMBLE_SEED,
    radius_px: float = ENSEMBLE_RADIUS_PX,
) -> list[dict[str, Any]]:
    """Apply the configured point policy without exposing it to Qwen."""

    if point_mode not in POINT_MODES:
        raise ValueError(f"unsupported point mode: {point_mode!r}")
    x0, y0, x1, y1 = bbox_1000_to_pixels(bbox_2d_1000, width, height)
    image_x1 = float(max(int(width) - 1, 0))
    image_y1 = float(max(int(height) - 1, 0))
    x0, x1 = np.clip([x0, x1], 0.0, image_x1).tolist()
    y0, y1 = np.clip([y0, y1], 0.0, image_y1).tolist()
    center = np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float64)
    points = [center]
    offsets = [np.zeros((2,), dtype=np.float64)]
    if point_mode == "ensemble5":
        rng = np.random.default_rng(int(seed))
        angles = rng.uniform(0.0, 2.0 * math.pi, size=4)
        radii = float(radius_px) * np.sqrt(rng.uniform(0.0, 1.0, size=4))
        for angle, radius in zip(angles, radii, strict=True):
            requested = center + radius * np.asarray([math.cos(angle), math.sin(angle)])
            bounded = np.asarray(
                [np.clip(requested[0], x0, x1), np.clip(requested[1], y0, y1)],
                dtype=np.float64,
            )
            # Clamping toward a box containing the centroid cannot increase radius,
            # but retain an explicit guard for floating-point and future changes.
            delta = bounded - center
            norm = float(np.linalg.norm(delta))
            if norm > float(radius_px):
                bounded = center + delta * (float(radius_px) / norm)
            points.append(bounded)
            offsets.append(bounded - center)

    width_scale = float(max(int(width) - 1, 1))
    height_scale = float(max(int(height) - 1, 1))
    result = []
    for index, (point, offset) in enumerate(zip(points, offsets, strict=True)):
        result.append({
            "point_index": index,
            "pixel_uv": [float(point[0]), float(point[1])],
            "offset_px": [float(offset[0]), float(offset[1])],
            "d4rt_uv_norm": [float(point[0] / width_scale), float(point[1] / height_scale)],
        })
    return result


def _numeric_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.dtype.kind not in "bfiu" or not np.isfinite(array).all():
        raise ValueError("math input contains non-finite or non-numeric values")
    if array.size > 4096:
        raise ValueError("math input is too large")
    return array


def _distance(a: Any, b: Any) -> float:
    return float(np.linalg.norm(_numeric_array(a) - _numeric_array(b)))


def _norm(a: Any) -> float:
    return float(np.linalg.norm(_numeric_array(a)))


def _mean(a: Any) -> float:
    return float(np.mean(_numeric_array(a)))


def _std(a: Any) -> float:
    return float(np.std(_numeric_array(a)))


def _finite_sum(a: Any) -> float:
    return float(np.sum(_numeric_array(a)))


def _visible_path_length(points: Any, visible: Any | None = None) -> float:
    xyz = _numeric_array(points)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("path_length points must have shape [N,3]")
    if visible is None:
        mask = np.ones((xyz.shape[0],), dtype=bool)
    else:
        mask = np.asarray(visible, dtype=bool)
        if mask.shape != (xyz.shape[0],):
            raise ValueError("path_length visibility must have shape [N]")
    total = 0.0
    previous: np.ndarray | None = None
    for point, is_visible in zip(xyz, mask, strict=True):
        if not bool(is_visible):
            previous = None
            continue
        if previous is not None:
            total += float(np.linalg.norm(point - previous))
        previous = point
    return total


SAFE_MATH_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": lambda value: float(np.min(_numeric_array(value))),
    "max": lambda value: float(np.max(_numeric_array(value))),
    "sum": _finite_sum,
    "round": round,
    "sqrt": math.sqrt,
    "hypot": math.hypot,
    "dist": _distance,
    "norm": _norm,
    "mean": _mean,
    "std": _std,
    "path_length": _visible_path_length,
}


class _RestrictedMathEvaluator:
    """Small AST interpreter; it never calls ``eval`` or ``exec``."""

    _binary = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
        ast.Mod: lambda a, b: a % b,
    }
    _unary = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}

    def __init__(self, bindings: Mapping[str, Any]) -> None:
        self.environment: dict[str, Any] = {}
        for name, value in bindings.items():
            if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
                raise ValueError(f"invalid math binding name: {name!r}")
            array = _numeric_array(value)
            self.environment[name] = float(array) if array.ndim == 0 else array
        self.outputs: dict[str, Any] = {}

    def run(self, code: str) -> dict[str, Any]:
        if len(code) > 4096:
            raise ValueError("math code exceeds 4096 characters")
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as error:
            raise ValueError(f"invalid math syntax: {error.msg}") from error
        if len(tree.body) > 32:
            raise ValueError("math code exceeds 32 statements")
        for statement in tree.body:
            if isinstance(statement, ast.Assign):
                if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                    raise ValueError("math assignments require one plain variable name")
                name = statement.targets[0].id
                if name.startswith("_") or not name.isidentifier():
                    raise ValueError(f"invalid assignment name: {name!r}")
                value = self._expression(statement.value)
                self._check_finite(value)
                self.environment[name] = value
                self.outputs[name] = self._jsonable(value)
            elif isinstance(statement, ast.Expr):
                value = self._expression(statement.value)
                self._check_finite(value)
                self.outputs["result"] = self._jsonable(value)
            else:
                raise ValueError(
                    "only numeric assignments and expressions are permitted; "
                    f"rejected {type(statement).__name__}"
                )
        if not self.outputs:
            raise ValueError("math code produced no result")
        return self.outputs

    def _expression(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return bool(node.value)
            if not isinstance(node.value, (int, float)):
                raise ValueError("only numeric constants are permitted")
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in self.environment:
                raise ValueError(f"unknown math variable: {node.id}")
            return self.environment[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return _numeric_array([self._expression(item) for item in node.elts])
        if isinstance(node, ast.BinOp) and type(node.op) in self._binary:
            try:
                left = self._expression(node.left)
                right = self._expression(node.right)
                if isinstance(node.op, ast.Pow):
                    exponent = np.asarray(right, dtype=np.float64)
                    if exponent.size != 1 or abs(float(exponent)) > 16:
                        raise ValueError("power exponent must be a scalar with magnitude <= 16")
                return self._binary[type(node.op)](left, right)
            except (ArithmeticError, FloatingPointError, TypeError, ValueError) as error:
                raise ValueError(f"numeric arithmetic failed: {error}") from error
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._unary:
            return self._unary[type(node.op)](self._expression(node.operand))
        if isinstance(node, ast.Subscript):
            value = self._expression(node.value)
            index = self._index(node.slice)
            try:
                return value[index]
            except (IndexError, TypeError) as error:
                raise ValueError(f"invalid numeric indexing: {error}") from error
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_MATH_FUNCTIONS:
                raise ValueError("function is not in the safe finite-math allowlist")
            if node.keywords:
                raise ValueError("keyword arguments are not permitted")
            arguments = [self._expression(argument) for argument in node.args]
            try:
                return SAFE_MATH_FUNCTIONS[node.func.id](*arguments)
            except (ArithmeticError, TypeError, ValueError) as error:
                raise ValueError(f"safe math call failed: {error}") from error
        raise ValueError(f"unsafe or unsupported math syntax: {type(node).__name__}")

    def _index(self, node: ast.AST) -> int | slice:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self._expression(node.operand)
            if isinstance(value, int):
                return -value
        if isinstance(node, ast.Slice):
            values = []
            for part in (node.lower, node.upper, node.step):
                if part is None:
                    values.append(None)
                else:
                    value = self._expression(part)
                    if not isinstance(value, int):
                        raise ValueError("slice values must be integers")
                    values.append(value)
            return slice(*values)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            raise ValueError(
                "string-key indexing is not permitted in numeric code; bind each "
                "evidence field to its own variable, then use those numeric variables "
                "directly (for example, path_length(points, visibility))"
            )
        raise ValueError("indices must be integer constants or slices")

    @staticmethod
    def _check_finite(value: Any) -> None:
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or not np.isfinite(array.astype(np.float64)).all():
            raise ValueError("math result is non-numeric or non-finite")
        if array.size > 4096:
            raise ValueError("math result is too large")

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value


def restricted_python_math(bindings: Mapping[str, Any], code: str) -> dict[str, Any]:
    """Execute safe finite numerical expressions over already-resolved bindings."""

    return _RestrictedMathEvaluator(bindings).run(code)
