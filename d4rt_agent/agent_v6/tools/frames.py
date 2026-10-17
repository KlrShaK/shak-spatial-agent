"""Plain frame retrieval, independent of segmentation and scene geometry."""

from __future__ import annotations

from PIL import Image

from ..reports import Report


def get_frames(workspace, times=None, range=None):
    views = workspace.frames(times=times, range=range, limit=4)
    images = []
    for view in views:
        image = Image.fromarray(view.image).convert("RGB")
        image.thumbnail((640, 640))
        images.append((view.label, image))
    return Report("Frames", headline=f"{len(images)} plain frames.", images=images,
                  raw={"frame_indices": [view.index for view in views]})
