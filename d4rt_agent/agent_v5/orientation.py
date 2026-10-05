"""An entity's facing per frame, from Orient-Anything-V2, in the scene's axes.

The conversion is AgentV4's (``traj3d_analyze.build_orientation``): OAV2's azimuth / elevation /
roll on a mask crop -> the subject's axes in that frame's camera (``oav2_axes_in_opencv``), rotated
from the crop's implicit optical axis to the frame's (``offaxis_correction``), then into the world
with the frame's camera rotation. Only frames where OAV2 resolves a unique front (``alpha == 1``)
carry a facing; a symmetric or unresolvable object has none, and none is reported.

Bearings follow AgentV4's DSI-Bench convention: the camera's bearing in the subject's frame is
``atan2(right, forward)`` of the subject-to-camera vector -- 0 in front of the subject, +90 its
right, +-180 behind, -90 its left.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from d4rt_agent.sam_orientany_d4rt_test import traj3d_buckets, traj3d_geometry


def source_intrinsics(scene: Any, pos: int) -> tuple[float, float, float, float]:
    """``(fx, fy, cx, cy)`` of scene frame ``pos`` at the source image resolution."""

    K = scene.K[pos]
    sy = scene.source_hw[0] / scene.model_hw[0]
    sx = scene.source_hw[1] / scene.model_hw[1]
    return float(K[0, 0] * sx), float(K[1, 1] * sy), float(K[0, 2] * sx), float(K[1, 2] * sy)


def facing_rows(model: Any, clip: Any, scene: Any, masks_at: Any, positions: Sequence[int]) -> list[dict[str, Any]]:
    """One row per scene position: OAV2's angles and, when usable, the facing in world and levelled axes.

    ``masks_at(pos)`` returns the entity's mask on that scene frame (mask resolution) or None.
    """

    import cv2
    from d4rt_agent.sam_orientany_d4rt_test.traj3d_orient import crop_subject, predict_angles

    rows: list[dict[str, Any]] = []
    for pos in positions:
        pos = int(pos)
        row: dict[str, Any] = {"pos": pos, "usable": False}
        mask = masks_at(pos)
        if mask is None or not mask.any():
            row["status"] = "not visible"
            rows.append(row)
            continue
        frame = clip.frame(int(scene.gf_index[pos]))
        full = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]),
                          interpolation=cv2.INTER_NEAREST) > 0
        cropped = crop_subject(frame, full)
        if cropped is None:
            row["status"] = "empty crop"
            rows.append(row)
            continue
        image, centre = cropped
        try:
            angles = predict_angles(model, image)
        except Exception as error:  # noqa: BLE001 -- one bad frame is not fatal
            row["status"] = f"error: {error}"
            rows.append(row)
            continue
        forward, _, up = traj3d_geometry.oav2_axes_in_opencv(angles["azimuth_deg"], angles["elevation_deg"],
                                                             angles["roll_deg"])
        intr = source_intrinsics(scene, pos)
        forward = traj3d_geometry.offaxis_correction(forward, centre, intr)
        up = traj3d_geometry.offaxis_correction(up, centre, intr)
        world_f = scene.cam_R[pos] @ forward
        world_u = scene.cam_R[pos] @ up
        row.update({
            "status": "ok", **angles, "usable": bool(angles["alpha"] == 1.0),
            "forward_cam": forward, "forward_world": world_f, "up_world": world_u,
            "forward_level": scene.to_level(world_f),
        })
        rows.append(row)
    return rows


def facing_words(forward_cam: np.ndarray) -> str:
    """How the subject faces as the camera sees it: toward the camera, away, image-left/right."""

    x, _, z = (float(v) for v in forward_cam)
    if abs(z) >= abs(x):
        return "toward the camera" if z < 0 else "away from the camera"
    return "to the image right" if x > 0 else "to the image left"


def heading_deg(forward_level: np.ndarray) -> float:
    """Azimuth of a levelled direction about gravity: 0 = the first camera's forward, + = to its right."""

    return float(np.degrees(np.arctan2(forward_level[0], forward_level[2])))


def subject_basis(row: dict[str, Any]) -> np.ndarray | None:
    """Columns [subject-right, subject-down, subject-forward] in world coordinates, from one row."""

    if not row.get("usable"):
        return None
    return traj3d_buckets._orthonormal_basis(row["forward_world"], row["up_world"])  # noqa: SLF001


def camera_bearing(scene: Any, subject_xyz: np.ndarray, pos: int, basis: np.ndarray) -> float:
    """The camera's bearing in the subject's frame: 0 front, +90 its right, +-180 behind, -90 its left."""

    right, _, forward = (scene.cam_t[pos] - subject_xyz) @ basis
    return float(np.degrees(np.arctan2(right, forward)))


def sector(bearing: float) -> tuple[str, str, float]:
    """(quadrant, octant, degrees from the nearest quadrant boundary)."""

    quadrant, centre = traj3d_buckets.quadrant_sector(bearing)
    octant, _ = traj3d_buckets.octant_sector(bearing)
    margin = 45.0 - abs(traj3d_buckets.wrapped_angle_difference(bearing, centre))
    return quadrant, octant, float(margin)
