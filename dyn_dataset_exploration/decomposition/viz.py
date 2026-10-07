"""Plots for the notebook: the problem tree, and the picture behind each step.

Colour is meaning throughout the tree: grey = not yet solved, green = solved by a
measurement, amber = fell back to Qwen with the evidence, red = could not be solved, blue =
answered from the frames because nothing measurable decides it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

STATUS_COLOR = {
    "pending": "#d9d9d9", "solved": "#8fd19e", "fallback": "#f6c177", "failed": "#ee8b8b",
    "frames": "#9ec5f0", "info": "#ffffff",
}
STAGES = ("S1", "S2", "S3", "S4")


@dataclass
class Node:
    id: str
    label: str
    layer: int                     # 0 question, 1 S1, 2 S2, 3 S3, 4 S4, 5 leaf
    stage: str                     # which stage solves it: S1..S4 or "leaf"
    status: str = "pending"
    detail: str = ""


@dataclass
class ProblemTree:
    nodes: list[Node] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)

    def add(self, node: Node, parents: Sequence[str] = ()) -> None:
        self.nodes.append(node)
        self.edges += [(p, node.id) for p in parents]


def build_tree(solution: Any) -> ProblemTree:
    """The tree of one solved (or failed) question, with every node's final status."""

    program, decision = solution.program, solution.decision
    tree = ProblemTree()
    q = solution.entry["question"]
    tree.add(Node("q", _wrap(q, 34), 0, "S1", "info"))
    tree.add(Node("kind", f"kind\n{program.kind}", 1, "S1", "solved", f"answer form: {program.answer_form}"), ["q"])
    w = program.window
    wl = w.kind if w.kind == "whole" else f"{w.kind} {w.phase or ''} {w.t0 if w.t0 is not None else ''}{'-' + str(w.t1) if w.t1 is not None else ''}".strip()
    tree.add(Node("window", f"when\n{wl}", 1, "S1", "solved"), ["kind"])
    tree.add(Node("options", f"options\n{sum(o.atoms is not None for o in program.options)}/{len(program.options)} read", 1, "S1",
                  "solved" if all(o.atoms is not None for o in program.options) or program.route == "vlm_only" else "fallback"), ["kind"])
    if program.route == "vlm_only":
        tree.add(Node("leaf", f"LEAF {decision.leaf}\nanswer {solution.final}", 2, "leaf", "frames",
                      decision.reason), ["kind"])
        return tree

    tracks = solution.tracks
    for e in program.entities:
        if e.kind == "camera":
            tree.add(Node(f"ent:{e.id}", "who\ncamera", 2, "S2", "solved"), ["kind"])
            continue
        got = tracks.get(e.id)
        ok = bool(got)
        n = len(got or [])
        label = f"who\n{e.phrase or 'main subject'}" + (f"\n({e.kind}, {n} track{'s' if n != 1 else ''})" if e.kind != "object" else "")
        tree.add(Node(f"ent:{e.id}", label, 2, "S2", "solved" if ok else "failed",
                      "" if ok else "the segmenter found nothing"), ["kind"])
    for ev in program.events:
        tree.add(Node(f"ev:{ev.id}", f"event\n{ev.predicate}", 3, "S2", "solved"),
                 [f"ent:{ev.actor}"] if any(e.id == ev.actor for e in program.entities) else ["kind"])
    ids = {m.id for m in program.measurements}
    first_col = 3 if program.events else 2
    depth: dict[str, int] = {}
    for m in program.measurements:
        deps = [depth[op] for op in m.of if op in depth]
        depth[m.id] = 1 + max(deps, default=0)
    for m in program.measurements:
        r = solution.results.get(m.id)
        status = "failed" if r is None or not r.available else "solved"
        parents = []
        for op in m.of:
            base = op.split("[")[0]
            if op in ids:
                parents.append(f"meas:{op}")
            elif op.startswith("ev"):
                parents.append(f"ev:{op}")
            else:
                parents.append(f"ent:{'cam' if base == 'camera' else base}")
        summary = _summary(m.signal, r)
        tree.add(Node(f"meas:{m.id}", f"{m.signal}\n{summary}", first_col + depth[m.id], "S3", status,
                      "" if r is None else (r.reason or "")), [p for p in parents if any(n.id == p for n in tree.nodes)] or ["kind"])
    last_col = first_col + max(depth.values(), default=0)
    last = f"meas:{program.measurements[-1].id}" if program.measurements else "kind"
    interp = "solved" if decision.scores else "failed"
    tree.add(Node("interp", f"score options\nmargin {decision.margin:.2f}", last_col + 1, "S4", interp, decision.reason), [last])
    leaf_status = {"A": "solved", "B": "fallback", "C": "frames"}[decision.leaf]
    tree.add(Node("leaf", f"LEAF {decision.leaf}\nanswer {solution.final}", last_col + 2, "leaf", leaf_status, decision.reason), ["interp"])
    return tree


def _summary(signal: str, r: Any) -> str:
    if r is None or not r.available:
        return "unavailable"
    v = r.value
    if signal == "camera_motion":
        return f"{v['motion']}" + (f" {v['direction']}" if v.get("direction") else "")
    if signal == "camera_jitter":
        return f"level {v['level']} ({v['rotation_residual_deg']:.2f} deg)"
    if signal in ("range", "region_distance"):
        return v.get("trend") or (f"{v['metres']:.2f} m" if "metres" in v else f"{v['units']:.3f} u")
    if signal == "displacement":
        return "path measured"
    if signal == "instances":
        return f"{v['count']} instance(s)"
    if signal == "onset":
        return f"frame {v['onset']}"
    if signal == "order":
        return f"first: {v['first']}"
    return ""


def _wrap(text: str, width: int) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines[:4]) + ("…" if len(lines) > 4 else "")


def draw_tree(tree: ProblemTree, upto: str | None = None, ax: Any = None, title: str | None = None) -> Any:
    """Draw the tree left to right; nodes of stages after ``upto`` are drawn pending (grey)."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    order = {"S1": 1, "S2": 2, "S3": 3, "S4": 4, "leaf": 5}
    cutoff = order.get(upto or "leaf", 5)
    by_layer: dict[int, list[Node]] = {}
    for n in tree.nodes:
        by_layer.setdefault(n.layer, []).append(n)
    columns = max(by_layer) + 1
    tallest = max(len(v) for v in by_layer.values())
    if ax is None:
        _, ax = plt.subplots(figsize=(2.9 * columns, 1.0 + 0.8 * tallest))
    pos: dict[str, tuple[float, float]] = {}
    for layer, nodes in by_layer.items():
        for i, n in enumerate(nodes):
            pos[n.id] = (layer * 3.0, (len(nodes) - 1) / 2 - i)
    for a, b in tree.edges:
        if a in pos and b in pos:
            ax.annotate("", xy=(pos[b][0] - 1.15, pos[b][1]), xytext=(pos[a][0] + 1.15, pos[a][1]),
                        arrowprops={"arrowstyle": "-|>", "color": "#777", "lw": 1.0, "shrinkA": 0, "shrinkB": 0})
    for n in tree.nodes:
        x, y = pos[n.id]
        done = n.layer == 0 or order[n.stage] <= cutoff
        color = STATUS_COLOR[n.status] if done else STATUS_COLOR["pending"]
        ax.add_patch(FancyBboxPatch((x - 1.1, y - 0.36), 2.2, 0.72, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=color, ec="#444", lw=1.0))
        ax.text(x, y, n.label if (done or n.layer == 0) else n.label.split("\n")[0], ha="center", va="center", fontsize=8)
    top = (tallest - 1) / 2 + 0.85
    names = {"S1": "S1 decompose", "S2": "S2 ground", "S3": "S3 measure", "S4": "S4 interpret", "leaf": "answer"}
    for stage, name in names.items():
        xs = [pos[n.id][0] for n in tree.nodes if n.stage == stage and n.layer > 0]
        if xs:
            ax.text((min(xs) + max(xs)) / 2, top, name, ha="center", fontsize=9, weight="bold", color="#333")
    ax.text(0, top, "question", ha="center", fontsize=9, weight="bold", color="#333")
    ax.set_xlim(-1.4, (columns - 1) * 3.0 + 1.4)
    ax.set_ylim(-tallest / 2 - 0.6, top + 0.4)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


# --------------------------------------------------------------------------- #
# frames, masks, timelines
# --------------------------------------------------------------------------- #

def frame_image(out: Path, slug: str, index: int) -> np.ndarray:
    from PIL import Image

    with Image.open(Path(out) / "clips" / slug / "frames" / f"{index:06d}.jpg") as im:
        return np.asarray(im.convert("RGB"))


def contact_sheet(out: Path, clip: dict[str, Any], indices: Sequence[int] | None = None, cols: int = 8,
                  title: str | None = None) -> Any:
    import matplotlib.pyplot as plt

    idx = list(clip["sampled16"] if indices is None else indices)
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 1.7 * rows))
    axes = np.atleast_1d(axes).ravel()
    t = clip.get("t_s")
    for ax, i in zip(axes, idx):
        ax.imshow(frame_image(out, clip["clip_slug"], i))
        ax.set_title(f"GF {i}" + (f" · {t[i]:.1f}s" if t else ""), fontsize=7)
        ax.axis("off")
    for ax in axes[len(idx):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def overlay_masks(image: np.ndarray, masks: Sequence[np.ndarray], colors: Sequence[str], alpha: float = 0.45) -> np.ndarray:
    import matplotlib.colors as mc

    out = image.astype(np.float32).copy()
    h, w = out.shape[:2]
    for mask, c in zip(masks, colors):
        if mask is None or not mask.any():
            continue
        if mask.shape != (h, w):
            ys = (np.arange(h) * mask.shape[0] / h).astype(int)
            xs = (np.arange(w) * mask.shape[1] / w).astype(int)
            mask = mask[np.ix_(ys, xs)]
        rgb = np.array(mc.to_rgb(c)) * 255
        out[mask] = (1 - alpha) * out[mask] + alpha * rgb
    return out.astype(np.uint8)


PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c"]


def mask_strip(out: Path, scene: Any, grounding: Any, chosen: dict[int, str], frames: Sequence[int] | None = None,
               title: str | None = None) -> Any:
    """Frames with every SAM3 instance overlaid; the chosen ones are drawn solid and labelled."""

    import matplotlib.pyplot as plt

    frames = list(frames if frames is not None else np.linspace(0, scene.length - 1, 8).round().astype(int))
    fig, axes = plt.subplots(1, len(frames), figsize=(2.3 * len(frames), 2.2))
    axes = np.atleast_1d(axes)
    for ax, pos in zip(axes, frames):
        gf = int(scene.gf_index[pos])
        img = frame_image(out, scene.clip["clip_slug"], gf)
        masks = [grounding.masks[k, gf] for k in range(grounding.count)]
        colors = [PALETTE[k % len(PALETTE)] for k in range(grounding.count)]
        ax.imshow(overlay_masks(img, masks, colors, alpha=0.5))
        ax.set_title(f"frame {gf}", fontsize=7)
        ax.axis("off")
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=PALETTE[k % len(PALETTE)],
                          label=f"#{k}" + (f" -> {chosen[k]}" if k in chosen else "")) for k in range(grounding.count)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), fontsize=7, frameon=False)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return fig


def lifespans(grounding: Any, scene: Any, chosen: dict[int, str], ax: Any = None) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.5 + 0.35 * max(grounding.count, 1)))
    frames = scene.gf_index
    for k in range(grounding.count):
        pres = grounding.present(k)[frames]
        ax.fill_between(np.arange(len(frames)), k - 0.35, k + 0.35, where=pres, color=PALETTE[k % len(PALETTE)], alpha=0.8, step="mid")
        ax.text(len(frames) + 0.5, k, chosen.get(k, ""), va="center", fontsize=8)
    ax.set_yticks(range(grounding.count), [f"#{k}" for k in range(grounding.count)])
    ax.set_xlabel("scene frame")
    ax.set_title(f"SAM3 instances of {grounding.phrase!r}: when each is present", fontsize=9)
    ax.set_xlim(-0.5, len(frames) + 6)
    ax.invert_yaxis()
    return ax


def timeline(scene: Any, clip: dict[str, Any], window: tuple[int, int] | None, note: str = "", ax: Any = None) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 1.6))
    T = scene.length
    ax.plot([0, T - 1], [0, 0], color="#999", lw=1)
    ax.scatter(np.searchsorted(scene.gf_index, clip["sampled16"]), np.zeros(len(clip["sampled16"])), s=22, c="#4363d8", zorder=3,
               label="the 16 frames the benchmark samples")
    if window is not None:
        ax.axvspan(window[0], window[1], color="#3cb44b", alpha=0.3, label=f"window {window[0]}..{window[1]}")
    t = scene.t_s
    ticks = np.linspace(0, T - 1, 6).round().astype(int)
    ax.set_xticks(ticks, [f"{i}" + (f"\n{t[i]:.1f}s" if t is not None else "") for i in ticks], fontsize=7)
    ax.set_yticks([])
    ax.set_title(note, fontsize=8, loc="left")
    ax.legend(fontsize=7, loc="upper right", frameon=False, ncol=2)
    return ax


def camera_plot(scene: Any, state: dict[str, np.ndarray], result: Any | None = None) -> Any:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.2))
    x = np.arange(scene.length)
    axes[0].plot(x, np.degrees(np.unwrap(np.radians(state["pan"]))), label="pan (+ right)")
    axes[0].plot(x, state["tilt"], label="tilt (+ up)")
    axes[0].plot(x, state["roll"], label="roll", alpha=0.6)
    axes[0].set(title="camera rotation vs frame 0, levelled (deg)", xlabel="scene frame")
    axes[0].legend(fontsize=7)
    tr = state["trans_smooth"] - state["trans_smooth"][0]
    for k, name in enumerate(("right", "down", "forward")):
        axes[1].plot(x, tr[:, k] / scene.scale, label=name)
    axes[1].set(title="camera translation, levelled frame (scene scales)", xlabel="scene frame")
    axes[1].legend(fontsize=7)
    axes[2].plot(tr[:, 0] / scene.scale, tr[:, 2] / scene.scale, "-o", ms=2)
    axes[2].set(title="top-down camera path", xlabel="right", ylabel="forward", aspect="equal")
    if result is not None and result.available and result.window:
        for ax in axes[:2]:
            ax.axvspan(result.window[0], result.window[1], color="#3cb44b", alpha=0.12)
    fig = axes[0].figure
    fig.tight_layout()
    return fig


def scene_3d(scene: Any, tracks: dict[str, list[Any]], frame: int = 0, max_points: int = 2500) -> Any:
    """Depth cloud of one frame with the camera path and every entity path (top-down + side)."""

    import matplotlib.pyplot as plt

    cloud = scene.points[frame].reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(1)]
    cloud = cloud[:: max(1, len(cloud) // max_points)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (i, j, xl, yl) in zip(axes, ((0, 2, "right", "forward (depth)"), (2, 1, "forward (depth)", "down"))):
        ax.scatter(cloud[:, i], cloud[:, j], s=1, c="#bbbbbb")
        ax.plot(scene.cam_t[:, i], scene.cam_t[:, j], "-", color="#1f77b4", lw=2, label="camera")
        ax.scatter(*scene.cam_t[0, [i, j]], c="#1f77b4", marker="^", s=40)
        for k, (eid, lst) in enumerate(tracks.items()):
            for n, tr in enumerate(lst):
                ok = np.isfinite(tr.smooth).all(1)
                if ok.any():
                    c = PALETTE[(k + n) % len(PALETTE)]
                    ax.plot(tr.smooth[ok, i], tr.smooth[ok, j], "-", color=c, lw=2, label=tr.id)
                    first = np.flatnonzero(ok)[0]
                    ax.scatter(*tr.smooth[first, [i, j]], color=c, s=30, zorder=4)
        ax.set(xlabel=xl, ylabel=yl, title=f"depth cloud of frame {frame} + paths (world = first camera)")
        if yl == "down":
            ax.invert_yaxis()
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def series_plot(result: Any, ylabel: str, ax: Any = None, marks: dict[str, float] | None = None) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 2.8))
    s = result.series
    x = s.get("frames", np.arange(len(s["range"])))
    ax.plot(x, s["range"], ".", color="#999", ms=3, label="raw")
    if "smooth" in s:
        ax.plot(x, s["smooth"], "-", color="#d62728", lw=1.8, label="smoothed")
    tol = s.get("tol")
    if tol is not None and "smooth" in s and np.isfinite(s["smooth"]).any():
        ref = np.nanmedian(s["smooth"][np.isfinite(s["smooth"])][: max(1, int(np.isfinite(s["smooth"]).sum() // 5))])
        ax.axhspan(ref - tol, ref + tol, color="#1f77b4", alpha=0.12, label=f"tolerance ±{tol:.3f}")
    for name, xv in (marks or {}).items():
        ax.axvline(xv, ls="--", color="#444", lw=0.8)
        ax.text(xv, ax.get_ylim()[1], name, fontsize=7, va="top")
    if result.window:
        ax.axvspan(*result.window, color="#3cb44b", alpha=0.08)
    ax.set(xlabel="scene frame", ylabel=ylabel)
    ax.legend(fontsize=7)
    return ax


def option_bars(decision: Any, gt: str | None = None, ax: Any = None) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.5 + 0.55 * max(len(decision.scores), 1)))
    scores = decision.scores
    colors = {"match": "#3cb44b", "partial": "#f5b041", "mismatch": "#cccccc", "unverifiable": "#9ec5f0"}
    ax.barh([s.letter for s in scores], [s.score for s in scores], color=[colors[s.verdict] for s in scores])
    for i, s in enumerate(scores):
        ax.text(min(s.score + 0.02, 0.98), i, f"{s.text[:58]}", va="center", fontsize=7)
    ax.axvline(0.6, color="#444", ls=":", lw=0.8)
    ax.set_xlim(0, 1.6)
    ax.invert_yaxis()
    ax.set_xlabel("support from the measurement")
    if gt:
        ax.set_title(f"ground truth: {gt}", fontsize=8, loc="right")
    return ax


def md_table(rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None, max_width: int = 90) -> Any:
    """A list of dicts as a Markdown table (renders in every notebook frontend)."""

    from IPython.display import Markdown

    if not rows:
        return Markdown("*(empty)*")
    columns = list(columns or rows[0].keys())

    def cell(v: Any) -> str:
        text = "" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))
        text = text.replace("|", "\\|").replace("\n", " ")
        return text if len(text) <= max_width else text[: max_width - 1] + "…"

    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(cell(r.get(c)) for c in columns) + " |" for r in rows]
    return Markdown("\n".join(lines))
