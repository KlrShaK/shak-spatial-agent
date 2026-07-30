"""Isolated one-frame Qwen grounding for bbox and explicit-point requests.

This module deliberately does not know about orchestration evidence identifiers.
It validates one request, creates a fresh grounding-only message list, invokes an
already-loaded Qwen instance, and returns parsed geometry plus host-only
provenance.  The caller owns per-question caching and immutable ``qg_N`` IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Literal, Protocol
import unicodedata

from .simple_v2_contracts import NUM_SAMPLED_FRAMES, SampledVideo


GroundingMode = Literal["bbox", "points"]
GroundingCacheKey = tuple[str, int, str, str, int | None]

BBOX_MAX_NEW_TOKENS = 96
POINTS_MAX_NEW_TOKENS = 192
DEFAULT_GENERATION_SEED = 42
MAX_REQUEST_CHARACTERS = 500
MAX_POINT_COUNT = 8
MAX_DESCRIPTION_CHARACTERS = 200

PROMPT_PATH = Path(__file__).with_name("prompts") / "qwen_grounding_tool.md"
_WHITESPACE = re.compile(r"\s+")


class QwenGenerator(Protocol):
    """The small part of :class:`OfflineQwen` used by this tool."""

    model_path: str

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate one reply from a fresh message list."""


@dataclass(frozen=True)
class GroundingPoint:
    """One validated normalized image point."""

    xy: tuple[float, float]
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "xy": [float(self.xy[0]), float(self.xy[1])],
            "description": self.description,
        }


@dataclass(frozen=True)
class ParsedGrounding:
    """Immutable geometry parsed from one grounder reply."""

    mode: GroundingMode
    status: Literal["ok", "not_found"]
    bbox_2d_1000: tuple[float, float, float, float] | None = None
    points_2d_1000: tuple[GroundingPoint, ...] | None = None


class GroundingResponseError(ValueError):
    """Malformed Qwen reply with replayable raw response and provenance."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str,
        host_provenance: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.host_provenance = host_provenance


def normalize_grounding_request(request: str) -> str:
    """Return the exact NFKC/trim/whitespace/case-fold cache normalization."""

    if not isinstance(request, str):
        raise TypeError("grounding request must be a string")
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", request).strip()).casefold()


def _resolved_video_path(sampled_video: SampledVideo) -> str:
    return str(Path(sampled_video.video_path).expanduser().resolve())


def _sampled_mapping_digest(sampled_video: SampledVideo) -> str:
    encoded = json.dumps(
        [int(value) for value in sampled_video.original_indices],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sampled_video_clip_key(sampled_video: SampledVideo) -> str:
    """Return a stable digest binding a grounding to one decoded clip sample."""

    payload = {
        "resolved_video_path": _resolved_video_path(sampled_video),
        "total_original_frames": int(sampled_video.total_original_frames),
        "width": int(sampled_video.width),
        "height": int(sampled_video.height),
        "sampled_to_original": [
            int(value) for value in sampled_video.original_indices
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"clip_sha256:{hashlib.sha256(encoded).hexdigest()}"


def grounding_cache_key(
    sampled_video: SampledVideo,
    mode: str,
    t_src: int,
    request: str,
    count: int | None,
) -> GroundingCacheKey:
    """Build the hashable canonical key for one per-question grounding cache."""

    checked_mode, checked_t_src, checked_request, checked_count = _validate_request(
        sampled_video=sampled_video,
        mode=mode,
        t_src=t_src,
        request=request,
        count=count,
    )
    return (
        sampled_video_clip_key(sampled_video),
        checked_t_src,
        checked_mode,
        normalize_grounding_request(checked_request),
        checked_count,
    )


def grounding_cache_key_digest(cache_key: GroundingCacheKey) -> str:
    """Return a serializable provenance digest without exposing the cache key."""

    encoded = json.dumps(
        list(cache_key),
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_request(
    *,
    sampled_video: SampledVideo,
    mode: str,
    t_src: int,
    request: str,
    count: int | None,
) -> tuple[GroundingMode, int, str, int | None]:
    if mode not in {"bbox", "points"}:
        raise ValueError("grounding mode must be 'bbox' or 'points'")
    checked_mode: GroundingMode = mode
    if isinstance(t_src, bool) or not isinstance(t_src, int):
        raise ValueError("t_src must be an integer sampled-frame index")
    if not 0 <= t_src < NUM_SAMPLED_FRAMES:
        raise ValueError(f"t_src must be in [0,{NUM_SAMPLED_FRAMES - 1}]")
    frames = sampled_video.frames_rgb
    if getattr(frames, "ndim", None) != 4 or len(frames) != NUM_SAMPLED_FRAMES:
        raise ValueError(
            f"sampled_video must contain exactly {NUM_SAMPLED_FRAMES} RGB frames"
        )
    if len(sampled_video.original_indices) != NUM_SAMPLED_FRAMES:
        raise ValueError(
            "sampled_video must contain exactly "
            f"{NUM_SAMPLED_FRAMES} sampled-to-original indices"
        )
    if not isinstance(request, str):
        raise ValueError("grounding request must be a string")
    checked_request = request.strip()
    if not checked_request:
        raise ValueError("grounding request must be non-empty")
    if len(checked_request) > MAX_REQUEST_CHARACTERS:
        raise ValueError(
            f"grounding request must be at most {MAX_REQUEST_CHARACTERS} characters"
        )
    if checked_mode == "bbox":
        if count is not None:
            raise ValueError("count must be absent for bbox grounding")
        checked_count = None
    else:
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError("count is required as an integer for points grounding")
        if not 1 <= count <= MAX_POINT_COUNT:
            raise ValueError(f"points count must be in [1,{MAX_POINT_COUNT}]")
        checked_count = count
    return checked_mode, t_src, checked_request, checked_count


def _strict_json_field(raw: str, field: str) -> Any:
    if not isinstance(raw, str):
        raise TypeError("grounder response must be a string")
    decoder = json.JSONDecoder()
    for offset, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and set(value) == {field}:
            return value[field]
    raise ValueError(
        f"no strict JSON object containing exactly {field!r} was found"
    )


def _finite_number(value: Any, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must contain finite numbers, not booleans")
    normalized = float(value)
    if not 0.0 <= normalized <= 1000.0:
        raise ValueError(f"{field} coordinates must lie in [0,1000]")
    return normalized


def parse_bbox_response(
    raw: str,
) -> tuple[float, float, float, float] | None:
    """Parse one strict normalized bbox object from a Qwen reply."""

    value = _strict_json_field(raw, "bbox_2d_1000")
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox_2d_1000 must be null or a four-number list")
    box = tuple(
        _finite_number(item, field="bbox_2d_1000")
        for item in value
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("bbox_2d_1000 must have positive width and height")
    return box


def parse_points_response(
    raw: str,
    *,
    requested_count: int,
) -> tuple[GroundingPoint, ...] | None:
    """Parse up to ``requested_count`` strict normalized point objects."""

    if isinstance(requested_count, bool) or not isinstance(requested_count, int):
        raise ValueError("requested_count must be an integer")
    if not 1 <= requested_count <= MAX_POINT_COUNT:
        raise ValueError(f"requested_count must be in [1,{MAX_POINT_COUNT}]")
    value = _strict_json_field(raw, "points_2d_1000")
    if value is None or value == []:
        return None
    if not isinstance(value, list):
        raise ValueError("points_2d_1000 must be null or a list")
    if len(value) > requested_count:
        raise ValueError(
            "grounder returned more points than requested "
            f"({len(value)} > {requested_count})"
        )
    points: list[GroundingPoint] = []
    seen_xy: set[tuple[float, float]] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"xy", "description"}:
            raise ValueError(
                f"points_2d_1000[{index}] must contain exactly xy and description"
            )
        xy = item["xy"]
        if not isinstance(xy, list) or len(xy) != 2:
            raise ValueError(f"points_2d_1000[{index}].xy must be a two-number list")
        checked_xy = (
            _finite_number(xy[0], field=f"points_2d_1000[{index}].xy"),
            _finite_number(xy[1], field=f"points_2d_1000[{index}].xy"),
        )
        if checked_xy in seen_xy:
            raise ValueError("points_2d_1000 coordinates must be unique")
        seen_xy.add(checked_xy)
        description = item["description"]
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"points_2d_1000[{index}].description must be a non-empty string"
            )
        checked_description = description.strip()
        if len(checked_description) > MAX_DESCRIPTION_CHARACTERS:
            raise ValueError(
                f"points_2d_1000[{index}].description must be at most "
                f"{MAX_DESCRIPTION_CHARACTERS} characters"
            )
        points.append(
            GroundingPoint(xy=checked_xy, description=checked_description)
        )
    return tuple(points) or None


def parse_grounding_response(
    raw: str,
    *,
    mode: str,
    requested_count: int | None = None,
) -> ParsedGrounding:
    """Parse a bbox or points reply into an immutable normalized result."""

    if mode == "bbox":
        if requested_count is not None:
            raise ValueError("requested_count must be absent for bbox grounding")
        bbox = parse_bbox_response(raw)
        return ParsedGrounding(
            mode="bbox",
            status="not_found" if bbox is None else "ok",
            bbox_2d_1000=bbox,
        )
    if mode == "points":
        if requested_count is None:
            raise ValueError("requested_count is required for points grounding")
        points = parse_points_response(raw, requested_count=requested_count)
        return ParsedGrounding(
            mode="points",
            status="not_found" if points is None else "ok",
            points_2d_1000=points,
        )
    raise ValueError("grounding mode must be 'bbox' or 'points'")


def _bbox_user_prompt(request: str) -> str:
    return f"""Ground the target named {json.dumps(request, ensure_ascii=False)} in this image inside one tight bounding box.

Return exactly one JSON object and no other text:
{{"bbox_2d_1000":[x_min,y_min,x_max,y_max]}}

Normalize coordinates to integers from 0 to 1000 with origin at the top-left.
If the requested target is not visible, return:
{{"bbox_2d_1000":null}}"""


def _points_user_prompt(request: str, count: int) -> str:
    return f"""Ground up to {count} 2D point(s) for the request {json.dumps(request, ensure_ascii=False)} in this image.

Return exactly one JSON object and no other text:
{{"points_2d_1000":[{{"xy":[x,y],"description":"what this point marks"}}]}}

Normalize coordinates to integers from 0 to 1000 with origin at the top-left.
Return at most {count} points. Return fewer points rather than inventing unreliable
ones. If no requested point can be located, return:
{{"points_2d_1000":null}}"""


class QwenGroundingTool:
    """Run one isolated bbox/points request through an already-loaded Qwen."""

    def __init__(
        self,
        qwen: QwenGenerator,
        *,
        prompt_path: str | Path = PROMPT_PATH,
        seed: int = DEFAULT_GENERATION_SEED,
    ) -> None:
        self.qwen = qwen
        self.seed = int(seed)
        self.prompt_path = Path(prompt_path)
        self.system_prompt = self.prompt_path.read_text(encoding="utf-8").strip()
        if not self.system_prompt:
            raise ValueError(f"grounding system prompt is empty: {self.prompt_path}")
        self.prompt_version = hashlib.sha256(
            self.system_prompt.encode("utf-8")
        ).hexdigest()

    def ground(
        self,
        *,
        sampled_video: SampledVideo,
        mode: str,
        t_src: int,
        request: str,
        count: int | None = None,
    ) -> dict[str, Any]:
        """Ground one request and return parsed geometry plus host provenance."""

        checked_mode, checked_t_src, checked_request, checked_count = _validate_request(
            sampled_video=sampled_video,
            mode=mode,
            t_src=t_src,
            request=request,
            count=count,
        )
        cache_key = grounding_cache_key(
            sampled_video,
            checked_mode,
            checked_t_src,
            checked_request,
            checked_count,
        )
        max_new_tokens = (
            BBOX_MAX_NEW_TOKENS
            if checked_mode == "bbox"
            else POINTS_MAX_NEW_TOKENS
        )
        user_prompt = (
            _bbox_user_prompt(checked_request)
            if checked_mode == "bbox"
            else _points_user_prompt(checked_request, int(checked_count))
        )

        # Construct new containers and a new PIL image for every invocation.  No
        # main-agent message, DSI question, or previous grounding turn is reused.
        from PIL import Image

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.fromarray(
                            sampled_video.frames_rgb[checked_t_src]
                        ),
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        started = time.monotonic()
        raw = self.qwen.generate(
            messages,
            max_new_tokens=max_new_tokens,
        )
        wall_seconds = time.monotonic() - started
        provenance: dict[str, Any] = {
            "resolved_video_path": _resolved_video_path(sampled_video),
            "sampled_mapping_digest": _sampled_mapping_digest(sampled_video),
            "original_t_src": int(
                sampled_video.original_indices[checked_t_src]
            ),
            "clip_key": sampled_video_clip_key(sampled_video),
            "normalized_request": normalize_grounding_request(checked_request),
            "cache_key_digest": grounding_cache_key_digest(cache_key),
            "raw_grounder_response": raw,
            "grounding_system_prompt": self.system_prompt,
            "grounding_user_prompt": user_prompt,
            "grounding_prompt_version": self.prompt_version,
            "qwen_model_path": str(getattr(self.qwen, "model_path", "")),
            "generation_seed": int(getattr(self.qwen, "seed", self.seed)),
            "max_new_tokens": max_new_tokens,
            "inference_wall_time_seconds": wall_seconds,
            "cache_hit": False,
        }
        try:
            parsed = parse_grounding_response(
                raw,
                mode=checked_mode,
                requested_count=checked_count,
            )
        except (TypeError, ValueError) as error:
            raise GroundingResponseError(
                str(error),
                raw_response=raw,
                host_provenance=provenance,
            ) from error

        result: dict[str, Any] = {
            "status": parsed.status,
            "mode": parsed.mode,
            "t_src": checked_t_src,
            "request": checked_request,
        }
        if parsed.mode == "bbox":
            result["bbox_2d_1000"] = (
                [float(value) for value in parsed.bbox_2d_1000]
                if parsed.bbox_2d_1000 is not None
                else None
            )
        else:
            points = parsed.points_2d_1000 or ()
            result.update(
                requested_count=int(checked_count),
                returned_count=len(points),
                points_2d_1000=[point.as_dict() for point in points],
            )
        result["host_provenance"] = provenance
        return result
