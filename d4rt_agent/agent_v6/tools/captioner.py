"""Isolated, cached visual descriptions and Qwen's local video input adapter."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
from PIL import Image

from ..workspace import WorkspaceError

PROMPTS = Path(__file__).parent.parent / "prompts"
VIDEO_FPS = 2.0
VIDEO_MAX_PIXELS = 25_165_824  # 12,288 merged visual tokens at temporal patch size 2.


class VideoQwen:
    """Add video metadata to OfflineQwen without changing its V4/V5 callers.

    The base wrapper cannot forward processor kwargs. Keep its ordinary image/text
    path; video calls use the same local model, seed, and greedy generation with
    explicit source timestamps and a total pixel budget.
    """

    def __init__(self, qwen):
        self.qwen = qwen

    @property
    def processor(self):
        return self.qwen.processor

    def generate(self, messages, *, max_new_tokens=None, video_metadata=None):
        if video_metadata is None:
            return self.qwen.generate(messages, max_new_tokens=max_new_tokens)
        torch = self.qwen.torch
        torch.manual_seed(self.qwen.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.qwen.seed)
        # Installed Qwen smart_resize divides by the unpadded frame count.
        # Reserve the temporal padding cost before its resize calculation.
        counts = [len(item['frames_indices']) for item in video_metadata]
        pixel_budget = min(int(VIDEO_MAX_PIXELS * count / (2 * ((count + 1) // 2)))
                           for count in counts)
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            videos_kwargs={
                "video_metadata": video_metadata, "do_sample_frames": False,
                "size": {"shortest_edge": 128 * 32 * 32, "longest_edge": pixel_budget},
            },
        ).to(self.qwen.model.device)
        with torch.inference_mode():
            output = self.qwen.model.generate(
                **inputs,
                max_new_tokens=self.qwen.max_new_tokens if max_new_tokens is None else int(max_new_tokens),
                do_sample=False, num_beams=1,
            )
        return self.processor.batch_decode(
            output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def _processor(qwen):
    """Find the processor through the controller's counting wrapper."""
    while qwen is not None:
        processor = getattr(qwen, "processor", None)
        if processor is not None:
            return processor
        qwen = getattr(qwen, "qwen", None)
    return None


def video_input(workspace, qwen):
    """Sample before decoding, using Qwen's sampler and the clip's actual clock.

    Arrays are then submitted as one video with sampling disabled so the processor
    retains source indices. No scene or segmentation is needed. Unknown-fps clips
    use labelled images, because Qwen otherwise silently assumes 24 fps.
    """
    clip = workspace.clip
    if clip.fps is None:
        content = [{"type": "text", "text": "The frame rate is unknown. Refer only to the shown-frame labels."}]
        indices = clip.keyframes(8)
        for index in indices:
            image = Image.fromarray(clip.frame(int(index))).convert("RGB")
            image.thumbnail((640, 640))
            content.extend([{"type": "text", "text": workspace.mark_shown(int(index))},
                            {"type": "image", "image": image}])
        return content, None
    metadata = SimpleNamespace(total_num_frames=clip.length, fps=float(clip.fps),
                               duration=clip.length / float(clip.fps))
    processor = _processor(qwen)
    if processor is not None and getattr(processor, "video_processor", None) is not None:
        indices = processor.video_processor.sample_frames(metadata, fps=VIDEO_FPS)
    else:
        # Model-free fixtures follow the installed Qwen3-VL sampler exactly.
        count = min(max(int(clip.length / float(clip.fps) * VIDEO_FPS), 4), 768, clip.length)
        indices = np.linspace(0, clip.length - 1, count).round().astype(int)
    indices = np.asarray(indices, dtype=int)
    frames = np.stack([clip.frame(int(index)) for index in indices])
    details = {"total_num_frames": clip.length, "fps": float(clip.fps),
               "duration": metadata.duration, "frames_indices": indices.tolist()}
    return [{"type": "video", "video": frames}], [details]


def caption_clip(workspace, qwen) -> str:
    """One question-blind caption per clip, reused across all its questions."""
    instruction = (PROMPTS / "captioner.md").read_text().strip()

    cache_root = getattr(workspace.models, "cache", None)
    target = None if cache_root is None else Path(cache_root) / "clips" / workspace.clip.slug / "v6_caption.json"
    signature = hashlib.sha256(json.dumps({
        "prompt": instruction, "model": getattr(workspace.models, "qwen_model", None),
        "seed": getattr(workspace.models, "seed", None), "fps": workspace.clip.fps,
        "frames": workspace.clip.length, "sample_fps": VIDEO_FPS,
        "pixel_budget": VIDEO_MAX_PIXELS, "unknown_fps_keyframes": 8,
    }, sort_keys=True).encode()).hexdigest()

    def describe():
        if target is not None and target.exists():
            try:
                saved = json.loads(target.read_text())
                if saved.get("signature") == signature and isinstance(saved.get("caption"), str) and saved["caption"].strip():
                    return saved["caption"]
            except (OSError, ValueError, AttributeError):
                pass  # Interrupted or corrupt cache entries are recomputed.
        content, metadata = video_input(workspace, qwen)
        content.append({"type": "text", "text": "Describe this clip."})
        messages = [{"role": "system", "content": instruction}, {"role": "user", "content": content}]
        kwargs = {"max_new_tokens": 160}
        if metadata is not None:
            kwargs["video_metadata"] = metadata
        caption = qwen.generate(messages, **kwargs).strip()
        if not caption:
            raise WorkspaceError("The captioner returned no description.")
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps({"signature": signature, "caption": caption}))
            os.replace(temporary, target)
        return caption

    return workspace.cached(("visual_caption", signature), (), describe)
