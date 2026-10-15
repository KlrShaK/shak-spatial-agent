"""Align a newly queried sequence to the existing scene using the shared anchor image."""

import numpy as np


def align_to_scene(tracks, source_cloud, target_cloud):
    source = np.asarray(source_cloud).reshape(-1, 3).astype(float)
    target = np.asarray(target_cloud).reshape(-1, 3).astype(float)
    if source.shape != target.shape:
        raise ValueError("reseeding requires the same anchor pixel grid in both reconstructions")
    ok = np.isfinite(source).all(1) & np.isfinite(target).all(1)
    if ok.sum() < 10:
        raise ValueError("not enough common anchor geometry to align reseeded tracks")
    source, target = source[ok], target[ok]
    a, b = source.mean(0), target.mean(0)
    x, y = source - a, target - b
    if np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
        raise ValueError("degenerate anchor geometry")
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    signs = np.ones(3)
    signs[-1] = np.linalg.det(u @ vt)
    rotation = (u * signs) @ vt
    scale = float((singular * signs).sum() / ((x * x).sum() / len(x)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid reseed scale")
    translation = b - scale * rotation @ a
    transformed = np.asarray(tracks) @ rotation.T * scale + translation
    residual = np.linalg.norm(source @ rotation.T * scale + translation - target, axis=1)
    return transformed, {"scale": scale, "median_alignment_residual": float(np.median(residual)),
                         "anchor_correspondences": int(ok.sum())}
