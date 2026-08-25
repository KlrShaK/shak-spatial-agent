"""Stage 4: draw the 3D trajectory with per-frame orientation triads.

CPU only; runs on a login node in seconds.  Follows the Agg + FigureCanvasAgg
pattern already used by vis/build_like_demo_for_worldtrack.py.

The plot has to carry three things at once -- where the subject went, which way
it faced, and how much the underlying points disagreed -- so the individual
tracks are drawn faintly behind the aggregate rather than hidden.  A clean line
with no visible spread would overstate how certain any of this is.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from d4rt_agent.sam_orientany_d4rt_test.traj3d_common import (  # noqa: E402
    Target,
    TARGETS,
    default_run_dir,
    read_json,
    video_paths,
)

START_COLOUR = "#00b050"
END_COLOUR = "#e02020"
AXIS_COLOURS = ("#d62728", "#2ca02c", "#1f77b4")  # forward, lateral, up
AXIS_LABELS = ("forward", "lateral", "up")

# Forward is the axis the whole orientation stage exists to produce -- the other
# two only fix the roll about it -- so it gets an arrowhead and extra length
# rather than being one indistinguishable stick of three.
FORWARD_SCALE = 1.6
ARROWHEAD_RATIO = 0.3


def _equalise_axes(ax: Any, points: np.ndarray) -> None:
    """Force a cubic bounding box.

    Matplotlib's 3D axes stretch each dimension independently, which silently
    distorts a trajectory's shape -- a straight run can look like a curve.
    """

    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        return
    centre = finite.mean(axis=0)
    radius = float(np.max(np.abs(finite - centre))) or 1.0
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


def _draw(ax: Any, data: dict[str, np.ndarray], *, title: str, show_tracks: bool = True) -> None:
    track = data["smoothed"]
    if not np.isfinite(track).any():
        track = data["ransac_median"]
    finite = np.isfinite(track).all(axis=1)
    if not finite.any():
        ax.set_title(f"{title}\n(no usable trajectory)")
        return

    # Faint per-point tracks: the spread the aggregate is summarising.
    if show_tracks and "xyz" in data:
        xyz, keep = data["xyz"], data["keep"]
        for index in range(xyz.shape[0]):
            if not keep[index]:
                continue
            point_track = xyz[index]
            ok = np.isfinite(point_track).all(axis=1)
            if ok.sum() > 1:
                ax.plot(
                    point_track[ok, 0], point_track[ok, 1], point_track[ok, 2],
                    color="#888888", alpha=0.16, linewidth=0.7, zorder=1,
                )

    # The aggregate, graded along time so direction of travel is readable.
    index = np.flatnonzero(finite)
    cmap = plt.get_cmap("viridis")
    for a, b in zip(index[:-1], index[1:]):
        ax.plot(
            track[[a, b], 0], track[[a, b], 1], track[[a, b], 2],
            color=cmap(a / max(len(track) - 1, 1)), linewidth=2.6, zorder=3,
        )
    ax.scatter(
        track[finite, 0], track[finite, 1], track[finite, 2],
        c=index, cmap="viridis", s=16, depthshade=False, zorder=4,
    )

    _draw_triads(ax, track, data, finite)

    first, last = index[0], index[-1]
    ax.scatter(*track[first], color=START_COLOUR, s=190, marker="o",
               edgecolors="black", linewidths=1.2, zorder=6)
    ax.scatter(*track[last], color=END_COLOUR, s=230, marker="X",
               edgecolors="black", linewidths=1.2, zorder=6)

    ax.set_xlabel("X  (right)")
    ax.set_ylabel("Y  (down)")
    ax.set_zlabel("Z  (forward)")
    ax.set_title(title)
    _equalise_axes(ax, track[finite])


def _draw_triads(ax: Any, track: np.ndarray, data: dict[str, np.ndarray], finite: np.ndarray) -> None:
    """Draw the subject's own axes at each frame.

    Ambiguous frames (OAV2 alpha != 1, i.e. the model reports no unique front)
    are drawn dashed only while they are the exception. On two of the three
    target clips they are the rule -- 26/32 and 32/32 -- and a plot full of faint
    dashed triads reads as "here is the orientation, slightly uncertain" when the
    honest statement is "there is no orientation here". Past half, they are
    dropped and the legend says how many were withheld.
    """

    forward = data.get("forward_cam0")
    up = data.get("up_cam0")
    usable = data.get("orientation_usable")
    if forward is None or len(forward) == 0:
        return

    drawable = [t for t in range(min(len(forward), len(track))) if finite[t]]
    if usable is not None:
        usable_count = sum(1 for t in drawable if t < len(usable) and bool(usable[t]))
    else:
        usable_count = len(drawable)
    skip_ambiguous = drawable and usable_count <= len(drawable) / 2

    span = float(np.nanmax(np.abs(track[finite] - np.nanmean(track[finite], axis=0)))) or 1.0
    # 32 frames x 3 axes is 96 segments over one polyline; at 0.28 they read as
    # the subject of the plot and the trajectory reads as background.
    length = 0.10 * span

    for t in range(min(len(forward), len(track))):
        if not finite[t] or not np.isfinite(forward[t]).all():
            continue
        origin = track[t]
        is_usable = bool(usable[t]) if usable is not None and t < len(usable) else True
        if skip_ambiguous and not is_usable:
            continue
        # Dashed/faint marks frames where OAV2 reported no unique front, so the
        # direction shown there is not meaningful.
        style = "-" if is_usable else ":"
        opacity = 1.0 if is_usable else 0.3

        norm = float(np.linalg.norm(forward[t]))
        if norm > 1e-9:
            direction = forward[t] / norm
            reach = length * FORWARD_SCALE
            tip = origin + direction * reach
            # Shaft and head drawn separately, rather than as one quiver: a 3D
            # quiver splits its own length between the two, so the head lands
            # partway along the bar instead of terminating it. Here the shaft
            # spans the full reach and the head occupies only its last stretch,
            # so the arrow ends exactly at the tip.
            ax.plot(
                [origin[0], tip[0]], [origin[1], tip[1]], [origin[2], tip[2]],
                color=AXIS_COLOURS[0], linewidth=2.4 if is_usable else 1.2,
                alpha=opacity, linestyle=style, zorder=6,
            )
            head = reach * ARROWHEAD_RATIO
            base = tip - direction * head
            ax.quiver(
                base[0], base[1], base[2],
                direction[0], direction[1], direction[2],
                length=head, arrow_length_ratio=1.0, pivot="tail",
                color=AXIS_COLOURS[0], linewidth=2.4 if is_usable else 1.2,
                alpha=opacity, zorder=7,
            )

        if up is None or not np.isfinite(up[t]).all():
            continue
        for axis_vec, colour in zip(
            (np.cross(up[t], forward[t]), up[t]), AXIS_COLOURS[1:]
        ):
            norm = float(np.linalg.norm(axis_vec))
            if norm < 1e-9:
                continue
            end = origin + axis_vec / norm * length
            ax.plot(
                [origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]],
                color=colour, linewidth=1.6 if is_usable else 0.9,
                alpha=opacity, linestyle=style, zorder=5,
            )


def _legend_handles(data: dict[str, np.ndarray]) -> list[Line2D]:
    handles = [
        Line2D([], [], color=START_COLOUR, marker="o", linestyle="", markersize=10,
               markeredgecolor="black", label="start (t=0)"),
        Line2D([], [], color=END_COLOUR, marker="X", linestyle="", markersize=11,
               markeredgecolor="black", label="end (t=31)"),
        Line2D([], [], color="#888888", alpha=0.5, label="individual point tracks"),
    ]
    handles += [
        Line2D([], [], color=AXIS_COLOURS[0], linewidth=2.4, marker=">",
               markersize=7, markevery=[-1],
               label=f"subject forward (\u00d7{FORWARD_SCALE:g})")
    ]
    handles += [
        Line2D([], [], color=colour, linewidth=1.6, label=f"subject {label}")
        for colour, label in zip(AXIS_COLOURS[1:], AXIS_LABELS[1:])
    ]
    usable = data.get("orientation_usable")
    if usable is not None and len(usable) and not np.all(usable):
        ambiguous = int(len(usable) - np.count_nonzero(usable))
        if np.count_nonzero(usable) <= len(usable) / 2:
            # Matches _draw_triads: past half they are withheld, not drawn faint.
            label = f"{ambiguous}/{len(usable)} frames unresolved (not drawn)"
            handles.append(Line2D([], [], color="none", label=label))
        else:
            handles.append(
                Line2D([], [], color="#666666", linestyle=":",
                       label=f"orientation ambiguous ({ambiguous}/{len(usable)})")
            )
    return handles


def load_data(paths: Any) -> dict[str, np.ndarray]:
    return dict(np.load(paths.analysis / "trajectory.npz"))


def plot_target(target: Target, run_dir: Path, *, orbit: bool = True) -> list[Path]:
    paths = video_paths(run_dir, target)
    data = load_data(paths)
    summary_file = paths.analysis / "summary.json"
    summary = read_json(summary_file) if summary_file.exists() else {}

    written: list[Path] = []

    figure = plt.figure(figsize=(13, 6.5), dpi=170)
    subtitle = (
        f"{summary.get('tracks_kept', '?')}/{summary.get('tracks_total', '?')} tracks kept"
        f"  ·  {summary.get('orientation', {}).get('frames_usable', '?')}/32 usable orientations"
    )
    figure.suptitle(
        f'"{target.subject}"  —  {target.relative_path}\n{subtitle}',
        fontsize=11,
    )

    ax_main = figure.add_subplot(1, 2, 1, projection="3d")
    _draw(ax_main, data, title="3D trajectory (frame-0 camera)")
    ax_top = figure.add_subplot(1, 2, 2, projection="3d")
    _draw(ax_top, data, title="top-down (ground plane, X-Z)", show_tracks=False)
    # elev=0/azim=-90 looks along the camera's Y axis, so the screen shows X
    # right and Z forward -- the ground plane. elev=88 would look down *Z*,
    # which in OpenCV coords is the optical axis, not "down": that gave an
    # X-Y plot labelled "top-down", which is the one projection a viewer is
    # guaranteed to misread as a bird's-eye path.
    ax_top.view_init(elev=0, azim=-90)

    figure.legend(
        handles=_legend_handles(data), loc="lower center", ncol=4, frameon=False, fontsize=8
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    destination = paths.plots / "trajectory.png"
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    written.append(destination)

    if orbit:
        written.append(_render_orbit(paths, data, target))
    return written


def _render_orbit(paths: Any, data: dict[str, np.ndarray], target: Target) -> Path:
    """A slow rotation so depth is readable; a static 3D plot hides it."""

    from PIL import Image

    frames = []
    for azim in range(0, 360, 8):
        figure = plt.figure(figsize=(6.5, 6.0), dpi=110)
        ax = figure.add_subplot(1, 1, 1, projection="3d")
        _draw(ax, data, title=f'"{target.subject}"')
        ax.view_init(elev=18, azim=azim)
        figure.tight_layout()
        figure.canvas.draw()
        frames.append(Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[..., :3]))
        plt.close(figure)

    destination = paths.plots / "trajectory_orbit.gif"
    frames[0].save(
        destination, save_all=True, append_images=frames[1:], duration=110, loop=0
    )
    return destination


def plot_tracks_overlay(target: Target, run_dir: Path) -> Path | None:
    """Draw the tracked 2D points on the frames, as a sanity check."""

    import cv2

    paths = video_paths(run_dir, target)
    subject_file = paths.d4rt / "subject.npz"
    if not subject_file.exists():
        return None
    subject = dict(np.load(subject_file))
    uv = subject["uv_2d"]  # (P, T, 2) normalised
    data = load_data(paths)
    keep = data["keep"]
    points = read_json(paths.points_json)
    width, height = points["width"], points["height"]

    tiles = []
    for t in range(0, uv.shape[1], 4):
        frame_file = paths.frames / f"{t:05d}.jpg"
        if not frame_file.exists():
            continue
        canvas = cv2.imread(str(frame_file))
        for index in range(uv.shape[0]):
            x = int(round(float(uv[index, t, 0]) * (width - 1)))
            y = int(round(float(uv[index, t, 1]) * (height - 1)))
            colour = (0, 220, 0) if keep[index] else (0, 0, 235)
            cv2.circle(canvas, (x, y), 2, colour, -1)
        cv2.putText(canvas, f"t={t:02d}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)

    if not tiles:
        return None
    per_row = 4
    rows = []
    for start in range(0, len(tiles), per_row):
        chunk = tiles[start : start + per_row]
        while len(chunk) < per_row:
            chunk.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(chunk))
    destination = paths.plots / "tracks_2d_overlay.jpg"
    cv2.imwrite(str(destination), np.vstack(rows))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--no-orbit", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir or default_run_dir()
    for target in TARGETS:
        if args.only and target.slug not in args.only:
            continue
        written = plot_target(target, run_dir, orbit=not args.no_orbit)
        overlay = plot_tracks_overlay(target, run_dir)
        if overlay:
            written.append(overlay)
        print(f"{target.slug}: " + ", ".join(str(p) for p in written), flush=True)


if __name__ == "__main__":
    main()
