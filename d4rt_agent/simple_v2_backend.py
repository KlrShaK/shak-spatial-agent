"""Cached live D4RT decoder backend for :mod:`d4rt_agent.simple_v2`.

This module never reads predicted tracks from a demo bundle. It loads the local
model/checkpoint, encodes the shared 32-frame clip once, and serves every agent
query from that cached video memory.
"""

from __future__ import annotations

import gc
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from .simple_v2_contracts import NUM_SAMPLED_FRAMES, SampledVideo, grounding_points


DEFAULT_D4RT_DIR = Path("checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG")
DEFAULT_D4RT_CONFIG = DEFAULT_D4RT_DIR / "model.yaml"
DEFAULT_D4RT_CHECKPOINT = DEFAULT_D4RT_DIR / "opend4rt.ckpt"

# How `benchmark_scale` was obtained.  The default describes the WorldTrack demo
# bundles, whose scale comes from ground truth.  Benchmarks without ground truth
# pass their own string rather than inherit a claim that is not true of them.
DEFAULT_ALIGNMENT_TYPE = "gt_derived_global_median_scale"

# Appended to every query result, so the frame a position lives in travels with
# the position itself.  The default forbids all cross-frame combination, which is
# right when every measurement is a displacement inside one frame.  Benchmarks
# that must compare distances taken from different vantage points supply the
# more precise scalar-versus-vector rule instead.
DEFAULT_CROSS_FRAME_NOTE = (
    "Do not combine these positions with positions from a query that used a "
    "different t_cam."
)


def _json_vector(array: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(array, dtype=np.float64).tolist()]


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class LiveD4RTBackend:
    """One live D4RT model and one immutable cached video encoding."""

    def __init__(
        self,
        *,
        sampled_video: SampledVideo,
        point_mode: str,
        benchmark_scale: float,
        model_config: str | Path = DEFAULT_D4RT_CONFIG,
        checkpoint: str | Path = DEFAULT_D4RT_CHECKPOINT,
        device: str = "cuda",
        dtype: str = "bfloat16",
        query_chunk_size: int = 256,
        alignment_type: str = DEFAULT_ALIGNMENT_TYPE,
        cross_frame_note: str = DEFAULT_CROSS_FRAME_NOTE,
    ) -> None:
        import torch

        from infer_track_3d import _resolve_device, _resize_video, _unwrap_state_dict
        from src.core import load_checkpoint, load_yaml_config, seed_everything
        from src.eval.tasks import _encode_model_memory, _model_clip_frames
        from src.model import build_model

        if sampled_video.frames_rgb.shape[0] != NUM_SAMPLED_FRAMES:
            raise ValueError("live D4RT backend requires exactly 32 sampled frames")
        if not math.isfinite(float(benchmark_scale)) or float(benchmark_scale) <= 0:
            raise ValueError("benchmark scale must be finite and positive")

        self.torch = torch
        self._run_model_for_queries = self._import_query_runner()
        # Held on the instance so rebind_video can re-encode a new clip without
        # re-importing, and without reloading the 14 GB checkpoint.
        self._resize_video = _resize_video
        self._encode_model_memory = _encode_model_memory
        self.point_mode = point_mode
        self.benchmark_scale = float(benchmark_scale)
        self.alignment_type = str(alignment_type)
        self.cross_frame_note = str(cross_frame_note)
        self.model_config_path = Path(model_config)
        self.checkpoint_path = Path(checkpoint)
        self.query_chunk_size = max(1, int(query_chunk_size))
        if not self.model_config_path.exists():
            raise FileNotFoundError(f"D4RT model config not found: {self.model_config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"D4RT checkpoint not found: {self.checkpoint_path}")

        cfg = load_yaml_config(self.model_config_path)
        seed_everything(int(cfg.get_path("experiment.seed", 42)), deterministic=True)
        configured_frames = int(cfg.get_path("model.input.clip_frames", 0))
        if configured_frames != NUM_SAMPLED_FRAMES:
            raise ValueError(
                f"D4RT config must use {NUM_SAMPLED_FRAMES} frames, got {configured_frames}"
            )
        image_size = cfg.get_path("model.input.image_size", [256, 256])
        self.model_image_hw = (int(image_size[0]), int(image_size[1]))
        self.device = _resolve_device(device)
        if dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif dtype == "float16":
            self.dtype = torch.float16
        elif dtype == "float32":
            self.dtype = torch.float32
        else:
            raise ValueError(f"unsupported D4RT dtype: {dtype!r}")
        if self.device.type == "cpu" and self.dtype != torch.float32:
            self.dtype = torch.float32

        if self.device.type == "cuda":
            # Keep the 14 GB checkpoint and model state out of the 24 GiB host-RAM
            # allowance.  Peak GPU use while copying fp32 checkpoint tensors into
            # the configured model is comfortably below the requested 80 GB.
            previous_default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(self.dtype)
            try:
                with torch.device(self.device):
                    model = build_model(cfg["model"]).eval()
            finally:
                torch.set_default_dtype(previous_default_dtype)
            payload = torch.load(self.checkpoint_path, map_location=self.device)
        else:
            model = build_model(cfg["model"]).eval()
            payload = load_checkpoint(self.checkpoint_path, map_location="cpu")
        state_dict = _unwrap_state_dict(payload)
        if not state_dict:
            raise RuntimeError(f"no model weights found in checkpoint: {self.checkpoint_path}")
        load_result = model.load_state_dict(state_dict, strict=False)
        self.load_diagnostics = {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }
        del payload, state_dict
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            self.model = model.eval()
        else:
            self.model = model.to(device=self.device, dtype=self.dtype).eval()
        if int(_model_clip_frames(self.model)) != NUM_SAMPLED_FRAMES:
            raise RuntimeError("loaded D4RT model does not expose a 32-frame query embedding")

        self.encoding_count = 0
        self.rebind_video(sampled_video)

    def rebind_video(self, sampled_video: SampledVideo) -> None:
        """Point this backend at another clip, reusing the loaded model.

        Encoding a new clip costs seconds; loading the checkpoint costs minutes,
        so evaluating many videos in one process depends on this staying separate
        from ``__init__``.  The previous encoding is released before the next one
        is built, so peak memory never holds two.
        """

        if sampled_video.frames_rgb.shape[0] != NUM_SAMPLED_FRAMES:
            raise ValueError("live D4RT backend requires exactly 32 sampled frames")

        for attribute in ("memory", "video_tensor", "aspect_tensor"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        gc.collect()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()

        # Reassigned, not just re-encoded: query() reads width/height off this to
        # turn a [0,1000] box into pixels, and clips differ in resolution.
        self.sampled_video = sampled_video
        resized = self._resize_video(sampled_video.frames_rgb, image_hw=self.model_image_hw)
        self.video_tensor = (
            self.torch.from_numpy(resized)
            .to(device=self.device, dtype=self.dtype)
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            / 255.0
        )
        aspect_value = float(sampled_video.width) / float(max(1, sampled_video.height))
        self.aspect_tensor = self.torch.tensor(
            [[aspect_value]], device=self.device, dtype=self.dtype
        )
        with self.torch.inference_mode(), self._autocast_context():
            self.memory = self._encode_model_memory(
                model=self.model,
                video_b=self.video_tensor,
                aspect_b=self.aspect_tensor,
            )
        if self.memory is None:
            raise RuntimeError("D4RT model does not support cached video encoding")
        self.encoding_count += 1

    def _autocast_context(self):
        """Keep mixed-dtype Fourier/query layers valid during reduced-precision inference."""

        if self.device.type == "cuda" and self.dtype in {self.torch.bfloat16, self.torch.float16}:
            return self.torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    @staticmethod
    def _import_query_runner():
        from src.eval.tasks import _run_model_for_queries

        return _run_model_for_queries

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "live_d4rt_decoder",
            "point_mode": self.point_mode,
            "model_config": str(self.model_config_path),
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "model_image_hw": list(self.model_image_hw),
            "encoded_frames": NUM_SAMPLED_FRAMES,
            "encoding_count": self.encoding_count,
            "benchmark_alignment": {
                "type": self.alignment_type,
                "scale": self.benchmark_scale,
            },
            "checkpoint_load": self.load_diagnostics,
        }

    def gpu_memory(self) -> dict[str, Any]:
        if self.device.type != "cuda":
            return {"device": str(self.device)}
        index = self.device.index if self.device.index is not None else self.torch.cuda.current_device()
        properties = self.torch.cuda.get_device_properties(index)
        return {
            "name": properties.name,
            "total_gib": round(properties.total_memory / (1024**3), 3),
            "allocated_gib": round(self.torch.cuda.memory_allocated(index) / (1024**3), 3),
            "max_allocated_gib": round(self.torch.cuda.max_memory_allocated(index) / (1024**3), 3),
        }

    def query(
        self,
        *,
        label: str,
        bbox_2d_1000: list[float],
        t_src: int,
        t_tgt: list[int],
        t_cam: int,
    ) -> dict[str, Any]:
        """Decode one centroid or five-point policy for all requested targets."""

        if not (0 <= int(t_cam) < NUM_SAMPLED_FRAMES):
            raise ValueError("t_cam is outside [0,31]")
        if not (0 <= int(t_src) < NUM_SAMPLED_FRAMES):
            raise ValueError("t_src is outside [0,31]")
        targets = np.asarray(t_tgt, dtype=np.int64)
        if targets.ndim != 1 or targets.size == 0:
            raise ValueError("t_tgt must be a non-empty one-dimensional list")
        if np.any((targets < 0) | (targets >= NUM_SAMPLED_FRAMES)):
            raise ValueError("t_tgt contains a frame outside [0,31]")

        points = grounding_points(
            self.point_mode,
            bbox_2d_1000,
            self.sampled_video.width,
            self.sampled_video.height,
        )
        uv = np.asarray([point["d4rt_uv_norm"] for point in points], dtype=np.float32)
        num_points = uv.shape[0]
        num_targets = targets.shape[0]
        repeated_uv = np.repeat(uv, num_targets, axis=0)
        query = {
            "u": self.torch.from_numpy(repeated_uv[:, 0]).to(
                device=self.device, dtype=self.dtype
            ),
            "v": self.torch.from_numpy(repeated_uv[:, 1]).to(
                device=self.device, dtype=self.dtype
            ),
            "t_src": self.torch.full(
                (num_points * num_targets,), int(t_src), device=self.device, dtype=self.torch.long
            ),
            "t_tgt": self.torch.from_numpy(np.tile(targets, num_points)).to(
                device=self.device, dtype=self.torch.long
            ),
            "t_cam": self.torch.full(
                (num_points * num_targets,),
                int(t_cam),
                device=self.device,
                dtype=self.torch.long,
            ),
        }
        with self.torch.inference_mode(), self._autocast_context():
            outputs = self._run_model_for_queries(
                model=self.model,
                video_b=self.video_tensor,
                aspect_b=self.aspect_tensor,
                query=query,
                chunk_size=self.query_chunk_size,
                memory_b=self.memory,
            )

        reshaped: dict[str, np.ndarray] = {}
        for name, tensor in outputs.items():
            array = tensor.numpy()
            reshaped[name] = array.reshape(num_points, num_targets, *array.shape[1:])
        predictions = [
            self._aggregate_target(points, reshaped, target_offset, int(target))
            for target_offset, target in enumerate(targets.tolist())
        ]
        visibility = [bool(item["visible"]) for item in predictions]
        return {
            "backend": "live_d4rt_decoder",
            "point_mode": self.point_mode,
            "label": str(label),
            "bbox_2d_1000": [float(value) for value in bbox_2d_1000],
            "bbox_pixel": list(
                self._bbox_pixels(bbox_2d_1000)
            ),
            "sampled_to_original": self.sampled_video.mapping(),
            "t_src": int(t_src),
            "original_t_src": int(self.sampled_video.original_indices[int(t_src)]),
            "t_tgt": [int(value) for value in targets.tolist()],
            "original_t_tgt": [
                int(self.sampled_video.original_indices[int(value)]) for value in targets.tolist()
            ],
            "t_cam": int(t_cam),
            "original_t_cam": int(self.sampled_video.original_indices[int(t_cam)]),
            # Restated per result so the frame is visible where positions are used,
            # not only in the system prompt.
            "coordinate_frame": (
                f"Positions are expressed in the camera frame of sampled frame {int(t_cam)} "
                "(OpenCV axes: +x right, +y down, +z forward; origin at that camera). "
                f"{self.cross_frame_note}"
            ),
            "query_points": points,
            "predictions": predictions,
            "math_trajectory_aligned_xyz_m": [item["math_xyz_aligned_m"] for item in predictions],
            "math_visibility": visibility,
            "visibility_coverage": float(sum(visibility) / len(visibility)),
            "benchmark_alignment": {
                "type": self.alignment_type,
                "scale": self.benchmark_scale,
            },
        }

    def _bbox_pixels(self, bbox: list[float]) -> tuple[float, float, float, float]:
        from .simple_v2_contracts import bbox_1000_to_pixels

        return bbox_1000_to_pixels(bbox, self.sampled_video.width, self.sampled_video.height)

    def _aggregate_target(
        self,
        points: list[dict[str, Any]],
        output: dict[str, np.ndarray],
        target_offset: int,
        target: int,
    ) -> dict[str, Any]:
        individual: list[dict[str, Any]] = []
        for point_index, point in enumerate(points):
            xyz = np.asarray(output["xyz_3d"][point_index, target_offset], dtype=np.float64)
            uv = np.asarray(output["uv_2d"][point_index, target_offset], dtype=np.float64)
            visibility_logit = float(output["visibility"][point_index, target_offset])
            confidence = float(output["confidence"][point_index, target_offset])
            visible_probability = _sigmoid(visibility_logit) if math.isfinite(visibility_logit) else 0.0
            finite = bool(np.isfinite(xyz).all())
            valid = bool(finite and visible_probability > 0.5)
            individual.append({
                **point,
                "raw_xyz": _json_vector(xyz) if finite else None,
                "benchmark_aligned_xyz_m": _json_vector(xyz * self.benchmark_scale) if finite else None,
                "visibility_logit": visibility_logit if math.isfinite(visibility_logit) else None,
                "visibility_probability": visible_probability,
                "visible": bool(visible_probability > 0.5),
                "confidence": confidence if math.isfinite(confidence) else None,
                "confidence_probability": _sigmoid(confidence) if math.isfinite(confidence) else None,
                "target_uv_norm": _json_vector(uv) if np.isfinite(uv).all() else None,
                "target_uv_px": [
                    float(uv[0] * max(self.sampled_video.width - 1, 1)),
                    float(uv[1] * max(self.sampled_video.height - 1, 1)),
                ] if np.isfinite(uv).all() else None,
                "valid_for_policy_average": valid,
            })

        valid_items = [item for item in individual if item["valid_for_policy_average"]]
        required = 1 if self.point_mode == "centroid" else 3
        policy_valid = len(valid_items) >= required
        centroid_finite = self.point_mode == "centroid" and individual[0]["raw_xyz"] is not None
        output_available = policy_valid or centroid_finite
        if output_available:
            source_items = individual[:1] if self.point_mode == "centroid" else valid_items
            raw_stack = np.asarray([item["raw_xyz"] for item in source_items], dtype=np.float64)
            aligned_stack = np.asarray(
                [item["benchmark_aligned_xyz_m"] for item in source_items], dtype=np.float64
            )
            raw_mean = raw_stack.mean(axis=0)
            aligned_mean = aligned_stack.mean(axis=0)
            raw_std = raw_stack.std(axis=0)
            aligned_std = aligned_stack.std(axis=0)
            finite_uv = [item["target_uv_norm"] for item in source_items if item["target_uv_norm"] is not None]
            uv_mean = np.asarray(finite_uv, dtype=np.float64).mean(axis=0) if finite_uv else None
            finite_confidence = [float(item["confidence"]) for item in source_items if item["confidence"] is not None]
            confidence = float(np.mean(finite_confidence)) if finite_confidence else None
        else:
            raw_mean = aligned_mean = raw_std = aligned_std = uv_mean = None
            confidence = None

        aligned_json = _json_vector(aligned_mean) if aligned_mean is not None else None
        return {
            "sampled_frame_index": int(target),
            "original_frame_index": int(self.sampled_video.original_indices[int(target)]),
            "raw_xyz": _json_vector(raw_mean) if raw_mean is not None else None,
            "benchmark_aligned_xyz_m": aligned_json,
            "raw_xyz_std": _json_vector(raw_std) if raw_std is not None else None,
            "benchmark_aligned_xyz_std_m": _json_vector(aligned_std) if aligned_std is not None else None,
            "valid_count": len(valid_items),
            "required_valid_count": required,
            "visible": policy_valid,
            "confidence": confidence,
            "target_uv_norm": _json_vector(uv_mean) if uv_mean is not None else None,
            "target_uv_px": [
                float(uv_mean[0] * max(self.sampled_video.width - 1, 1)),
                float(uv_mean[1] * max(self.sampled_video.height - 1, 1)),
            ] if uv_mean is not None else None,
            "individual_point_results": individual,
            # Restricted math receives numeric arrays only. Invalid rows are zeroed
            # and must always be paired with math_visibility=false.
            "math_xyz_aligned_m": aligned_json if aligned_json is not None else [0.0, 0.0, 0.0],
        }
