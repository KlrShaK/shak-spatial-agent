"""The models one v5 job holds, each built on first use and then kept for every clip.

Qwen3-VL-8B (``simple_v2.OfflineQwen``), the SAM3 video predictor, the chosen backbone (and its
metric model), and Orient-Anything-V2. Lazy because a camera-only question never needs SAM3 or
OAV2, and a semantic one needs no backbone at all.

SAM3 groundings are cached per (clip, phrase, prompt frame) under
``<cache>/clips/<slug>/sam3/``, in the research pipeline's format, so a resumed run and a second
question on the same clip reuse them.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from d4rt_agent.agent_v5.backbones import SceneCache, get_backbone
from d4rt_agent.agent_v5.clips import Clip
from d4rt_agent.agent_v5.scene import Grounding

MASK_LONG_SIDE = 512


def phrase_key(phrase: str, prompt_frame: int = 0) -> str:
    """A filesystem-safe key for one SAM3 prompt (hashed: phrases are model-authored)."""

    text = phrase.strip().lower()
    stem = "".join(ch if ch.isalnum() else "_" for ch in text)[:32].strip("_")
    suffix = f"_f{prompt_frame}" if prompt_frame else ""
    return f"{stem}_{hashlib.sha1(text.encode()).hexdigest()[:8]}{suffix}"


class Models:
    def __init__(self, *, qwen_model: str | None, backbone: str, cache: Path,
                 max_new_tokens: int = 512, seed: int = 42) -> None:
        self.qwen_model = qwen_model
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.cache = Path(cache)
        self.backbone_name = backbone
        self._qwen: Any = None
        self._sam3: Any = None
        self._orient: Any = None
        self._scenes: SceneCache | None = None

    # -- language / vision model -----------------------------------------------

    def qwen(self) -> Any:
        if self._qwen is None:
            from d4rt_agent.simple_v2 import OfflineQwen

            self._qwen = OfflineQwen(self.qwen_model, max_new_tokens=self.max_new_tokens, seed=self.seed)
        return self._qwen

    # -- 4D backbone ---------------------------------------------------------------

    def scenes(self) -> SceneCache:
        if self._scenes is None:
            self._scenes = SceneCache(get_backbone(self.backbone_name), self.cache)
        return self._scenes

    # -- SAM3 ----------------------------------------------------------------------

    def sam3(self) -> Any:
        if self._sam3 is None:
            import torch
            from sam3.model_builder import build_sam3_video_predictor

            self._sam3 = build_sam3_video_predictor(gpus_to_use=range(max(torch.cuda.device_count(), 1)))
        return self._sam3

    def ground(self, clip: Clip, phrase: str, prompt_frame: int = 0) -> Grounding:
        """Every instance SAM3 finds for ``phrase`` (prompted on ``prompt_frame``, propagated both
        ways), on every GF frame. Cached."""

        target = self.cache / "clips" / clip.slug / "sam3" / f"{phrase_key(phrase, prompt_frame)}.npz"
        if not target.exists():
            result = ground_phrase(self.sam3(), clip, phrase, prompt_frame)
            target.parent.mkdir(parents=True, exist_ok=True)
            masks = result.pop("masks")
            tmp = target.with_name(target.stem + ".tmp.npz")
            np.savez_compressed(tmp, masks_packed=np.packbits(masks, axis=-1), masks_shape=np.asarray(masks.shape),
                                **result)
            os.replace(tmp, target)
        with np.load(target) as z:
            shape = tuple(int(v) for v in z["masks_shape"])
            masks = np.unpackbits(z["masks_packed"], axis=-1)[..., : shape[-1]].astype(bool).reshape(shape)
            return Grounding(phrase, z["obj_ids"], masks, z["probs"], z["boxes"], z["area_frac"], z["border"])

    # -- Orient-Anything-V2 ------------------------------------------------------------

    def orient(self) -> Any:
        if self._orient is None:
            from d4rt_agent.sam_orientany_d4rt_test.traj3d_orient import load_model

            self._orient = load_model()
        return self._orient


def ground_phrase(predictor: Any, clip: Clip, phrase: str, prompt_frame: int = 0) -> dict[str, Any]:
    """SAM3 video grounding of one phrase over the clip's GF frames (``prep_qwen_sam3.ground_phrase``,
    with a choosable prompt frame so an entity that enters late can still be found)."""

    import cv2
    from d4rt_agent.sam_orientany_d4rt_test.traj3d_segment import _to_numpy, bf16_autocast

    n_frames = clip.length
    per_frame: dict[int, dict[int, tuple[np.ndarray, float]]] = {}
    with bf16_autocast():
        session = predictor.handle_request({"type": "start_session", "resource_path": str(clip.frames_dir)})
        sid = session["session_id"]
        predictor.handle_request({"type": "add_prompt", "session_id": sid,
                                  "frame_index": int(prompt_frame), "text": phrase})
        for response in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                                                         "propagation_direction": "both"}):
            outputs = response["outputs"]
            ids = np.asarray(_to_numpy(outputs["out_obj_ids"])).reshape(-1)
            probs = np.asarray(_to_numpy(outputs["out_probs"])).reshape(-1)
            masks = np.asarray(_to_numpy(outputs["out_binary_masks"]))
            per_frame[int(response["frame_index"])] = {
                int(obj): (np.asarray(masks[k]).squeeze().astype(bool), float(probs[k])) for k, obj in enumerate(ids)}
        predictor.handle_request({"type": "close_session", "session_id": sid})

    height, width = (int(v) for v in clip.record["source_hw"])
    scale = MASK_LONG_SIDE / max(height, width)
    mh, mw = max(1, round(height * scale)), max(1, round(width * scale))
    obj_ids = sorted({obj for frame in per_frame.values() for obj in frame})
    masks = np.zeros((len(obj_ids), n_frames, mh, mw), dtype=bool)
    probs = np.full((len(obj_ids), n_frames), np.nan, dtype=np.float32)
    boxes = np.full((len(obj_ids), n_frames, 4), np.nan, dtype=np.float32)
    area = np.zeros((len(obj_ids), n_frames), dtype=np.float32)
    border = np.zeros((len(obj_ids), n_frames), dtype=bool)
    for t, frame in per_frame.items():
        if not 0 <= t < n_frames:
            continue
        for k, obj in enumerate(obj_ids):
            if obj not in frame:
                continue
            mask, prob = frame[obj]
            small = cv2.resize(mask.astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST) > 0
            if not small.any():
                continue
            masks[k, t], probs[k, t] = small, prob
            ys, xs = np.nonzero(mask)
            mask_h, mask_w = mask.shape
            boxes[k, t] = (xs.min() / mask_w, ys.min() / mask_h, (xs.max() + 1) / mask_w, (ys.max() + 1) / mask_h)
            area[k, t] = mask.mean()
            border[k, t] = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    return {"obj_ids": np.asarray(obj_ids, dtype=np.int64), "masks": masks, "probs": probs,
            "boxes": boxes, "area_frac": area, "border": border}


def gpu_memory_report() -> dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            return {"peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    except Exception:  # noqa: BLE001 -- a report, never fatal
        pass
    return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    os.replace(tmp, path)
