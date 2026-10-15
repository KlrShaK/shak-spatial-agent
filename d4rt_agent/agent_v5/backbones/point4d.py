"""Point4D: chunked long-range tracking over up to 400 GF frames, with geometry.

Live port of ``dyn_dataset_exploration/decomposition/prep_point4d.py`` (``stage_scene`` and
``stage_metric``). World = the camera of the first GF frame (+X right, +Y down, +Z forward),
Point4D's model-relative units; metres come from DA3-Nested: the median over the answer frames of
median(DA3 metric depth / Point4D camera depth), pixels above either map's 95th percentile left out.

Needs ``point4d`` importable (``POINT4D_REPO`` on ``PYTHONPATH``; the slurm body sets it) and the
DA3-Nested snapshot in the offline HF cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from d4rt_agent.agent_v5.backbones.base import Backbone, camera_depth, model_hw, query_grid
from d4rt_agent.agent_v5.clips import MAX_GF_FRAMES, Clip

CHUNK_SIZE = 48
OVERLAP = 8
GEOMETRY_STRIDE = 4
DA3_NESTED = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DA3_LONG_SIDE = 504


class Point4DBackbone(Backbone):
    name = "point4d"

    def __init__(self) -> None:
        self._model: Any = None
        self._da3: Any = None

    def model(self) -> Any:
        if self._model is None:
            from point4d.loader import load_model

            self._model = load_model(device="cuda")
        return self._model

    def reconstruct(self, clip: Clip) -> dict[str, np.ndarray]:
        import torch
        from point4d.inference import LongTrackConfig, track_long_video

        frames, gf_index = clip.all_frames(MAX_GF_FRAMES)
        hw = model_hw(clip)
        cfg = LongTrackConfig(chunk_size=CHUNK_SIZE, overlap=OVERLAP, image_hw=hw, geometry_stride=GEOMETRY_STRIDE)
        grid_uv = query_grid(hw)
        with torch.inference_mode():
            traj, conf, geom = track_long_video(self.model(), frames, grid_uv, cfg, return_geometry=True)
        torch.cuda.empty_cache()
        return {
            "gf_index": gf_index, "model_hw": np.asarray(hw), "source_hw": np.asarray(clip.record["source_hw"]),
            "cam_R": np.asarray(geom["cam_R"]), "cam_t": np.asarray(geom["cam_t"]), "K": np.asarray(geom["intrinsics"]),
            "points": np.asarray(geom["points"]).astype(np.float16), "geometry_stride": np.int32(GEOMETRY_STRIDE),
            "grid_uv": grid_uv, "grid_traj": np.asarray(traj).astype(np.float32),
            "grid_conf": np.asarray(conf).astype(np.float32),
        }

    # -- metric scale: DA3-Nested ------------------------------------------------

    def track_points(self, clip, scene, uv, anchor):
        import torch
        from point4d.inference import LongTrackConfig, track_long_video
        from d4rt_agent.agent_v5.backbones.reseed import align_to_scene

        result = np.full((scene.length, len(uv), 3), np.nan, dtype=np.float32)
        diagnostics = []
        cfg = LongTrackConfig(chunk_size=CHUNK_SIZE, overlap=OVERLAP,
                              image_hw=scene.model_hw, geometry_stride=scene.stride)
        # Each invocation starts at the chosen anchor; backwards processing is restored
        # to chronological positions after alignment to the existing scene's anchor cloud.
        for indices in (np.arange(anchor, scene.length), np.arange(anchor, -1, -1)):
            if len(indices) < 2:
                continue
            frames = clip.frames(scene.gf_index[indices].tolist())
            with torch.inference_mode():
                tracks, confidence, geometry = track_long_video(self.model(), frames, uv, cfg, return_geometry=True)
            aligned, quality = align_to_scene(tracks, geometry["points"][0], scene.points[anchor])
            if np.isfinite(result[anchor]).any():
                quality["anchor_disagreement"] = float(np.nanmedian(np.linalg.norm(result[anchor] - aligned[0], axis=1)))
                result[indices[1:]] = aligned[1:]
            else:
                result[indices] = aligned
            diagnostics.append({"scene_indices": indices.tolist(), "generator_frames": scene.gf_index[indices].tolist(), **quality})
        return result, diagnostics

    def da3(self) -> Any:
        if self._da3 is None:
            self._da3 = load_da3_nested()
        return self._da3

    def metric(self, clip: Clip, arrays: Mapping[str, np.ndarray]) -> dict[str, Any] | None:
        import cv2
        import torch

        gf_index = np.asarray(arrays["gf_index"])
        depth = camera_depth(np.asarray(arrays["points"]).astype(np.float32), np.asarray(arrays["cam_R"]),
                             np.asarray(arrays["cam_t"]))
        positions = [int(np.argmin(np.abs(gf_index - i))) for i in clip.answer_frames]
        frames = clip.frames([int(gf_index[p]) for p in positions])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.da3()(da3_input(frames).to("cuda"))
        metric = output.depth[0].float().cpu().numpy()          # (N, h, w) metres
        per_frame = []
        for n, position in enumerate(positions):
            p4d = depth[position]
            m = cv2.resize(metric[n], (p4d.shape[1], p4d.shape[0]), interpolation=cv2.INTER_AREA)
            ok = np.isfinite(p4d) & (p4d > 1e-6) & np.isfinite(m) & (m > 0)
            if ok.sum() < 10:
                continue
            ok &= (m < np.percentile(m[ok], 95)) & (p4d < np.percentile(p4d[ok], 95))  # sky, far
            per_frame.append(float(np.median(m[ok] / p4d[ok])))
        torch.cuda.empty_cache()
        if not per_frame:
            return None
        q25, q50, q75 = np.percentile(per_frame, [25, 50, 75])
        return {"metres_per_unit": float(q50), "iqr_over_median": float((q75 - q25) / max(q50, 1e-9)),
                "source": "da3-nested", "model": DA3_NESTED, "frames": len(per_frame)}


def load_da3_nested(device: str = "cuda") -> Any:
    """DA3-Nested through the DA3 code vendored in Point4D (the network definition is all that is
    needed; the official package pulls xformers/open3d). Built on the GPU: initialising 1.4B
    parameters on the CPU takes minutes and the checkpoint overwrites them anyway."""

    import torch
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from point4d.backbone.depth_anything_3.cfg import create_object

    root = Path(snapshot_download(DA3_NESTED, local_files_only=True))
    text = (root / "config.json").read_text().replace('"depth_anything_3.', '"point4d.backbone.depth_anything_3.')
    config = json.loads(text)["config"]
    for key in ("gs_head", "gs_adapter"):           # the Gaussian-splatting head is not vendored
        config["anyview"].pop(key, None)
    with torch.device(device):
        model = create_object(OmegaConf.create(config))
    state = {k[len("model."):]: v for k, v in load_file(str(root / "model.safetensors")).items()
             if k.startswith("model.") and ".gs_" not in k}
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


def da3_input(frames: np.ndarray) -> Any:
    """(1, N, 3, H, W): long side 504, sides multiples of 14, ImageNet-normalised."""

    import cv2
    import torch

    h, w = frames.shape[1:3]
    scale = DA3_LONG_SIDE / max(h, w)
    nh, nw = max(14, round(h * scale / 14) * 14), max(14, round(w * scale / 14) * 14)
    resized = np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in frames])
    x = torch.from_numpy(resized).float().permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)
