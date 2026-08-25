"""Orchestrate the traj3d pipeline over every target video.

Stages are ordered by *model*, not by video: all segmentation, then all D4RT,
then all orientation.  Each of the three checkpoints is therefore loaded once
per job rather than once per video, which matters most for D4RT (14 GB).

Every stage records its own completion on disk, so a job killed by the SLURM
time limit is continued by resubmitting the identical command -- the work
already done is skipped rather than repeated.
"""

from __future__ import annotations

import argparse
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from d4rt_agent.sam_orientany_d4rt_test.traj3d_common import (
    Target,
    TARGETS,
    default_run_dir,
    run_metadata,
    video_paths,
    write_json,
)

STAGES = ("segment", "query", "orient", "analyze")

# Per-video failures, keyed "<stage>/<slug>".  Populated by _isolated and written
# into run_status.json so a partial run says exactly what it dropped.
TARGET_ERRORS: dict[str, str] = {}


@contextmanager
def _isolated(stage: str, slug: str):
    """Contain one video's failure so it does not cost the other two.

    The videos are independent -- different clips, different subjects, no shared
    state beyond the loaded checkpoint -- so a phrase that grounds badly on one
    of them is no reason to abandon the rest of a multi-hour GPU job.
    """

    try:
        yield
    except Exception:
        TARGET_ERRORS[f"{stage}/{slug}"] = traceback.format_exc()
        print(
            f"[{stage}] {slug}: FAILED, continuing with the remaining targets\n"
            f"{TARGET_ERRORS[f'{stage}/{slug}']}",
            flush=True,
        )


def _all_failed(stage: str, attempted: list[str], succeeded: dict[str, Any]) -> None:
    """Raise only when nothing at all worked -- otherwise a partial run is fine."""

    if attempted and not succeeded:
        raise RuntimeError(f"stage {stage}: every target failed ({len(attempted)} of them)")


def _selected(only: list[str] | None) -> list[Target]:
    if not only:
        return list(TARGETS)
    chosen = [t for t in TARGETS if t.slug in only or t.relative_path in only]
    if not chosen:
        raise SystemExit(f"no target matched {only}")
    return chosen


def _done(target: Target, run_dir: Path, stage: str) -> bool:
    paths = video_paths(run_dir, target)
    marker = {
        "segment": paths.points_json,
        "query": paths.d4rt / "queries.json",
        "orient": paths.orientation / "orientation.json",
        "analyze": paths.analysis / "summary.json",
    }[stage]
    return marker.exists()


def run_stage_segment(targets: list[Target], run_dir: Path, force: bool) -> dict[str, Any]:
    from d4rt_agent.sam_orientany_d4rt_test.traj3d_segment import release_image_processor, segment_target

    results: dict[str, Any] = {}
    attempted: list[str] = []
    try:
        for target in targets:
            if not force and _done(target, run_dir, "segment"):
                print(f"[segment] {target.slug}: already done, skipping", flush=True)
                continue
            attempted.append(target.slug)
            with _isolated("segment", target.slug):
                record = segment_target(target, run_dir)
                results[target.slug] = record
                print(
                    f"[segment] {target.slug}: instance #{record['chosen_index']} "
                    f"(score {record['chosen_score']:.3f} of {record['num_candidates']}) "
                    f"on anchor frame t={record['anchor_frame']}, "
                    f"{len(record['repaired_frames'])} frames repaired, "
                    f"{len(record['frames_without_mask'])} without mask",
                    flush=True,
                )
    finally:
        # The SAM3 image model is cached across videos on purpose; drop it here
        # so D4RT's 14 GB checkpoint does not have to share with it.
        release_image_processor()
    _all_failed("segment", attempted, results)
    return results


def run_stage_query(targets: list[Target], run_dir: Path, force: bool) -> dict[str, Any]:
    from d4rt_agent.sam_orientany_d4rt_test.traj3d_query import query_target

    holder: dict[str, Any] = {}
    results: dict[str, Any] = {}
    attempted: list[str] = []
    for target in targets:
        if not force and _done(target, run_dir, "query"):
            print(f"[query] {target.slug}: already done, skipping", flush=True)
            continue
        attempted.append(target.slug)
        with _isolated("query", target.slug):
            record = query_target(target, run_dir, holder)
            results[target.slug] = record
            print(f"[query] {target.slug}: done in {record['wall_seconds']:.1f}s", flush=True)
    holder.pop("backend", None)
    _all_failed("query", attempted, results)
    return results


def run_stage_orient(targets: list[Target], run_dir: Path, force: bool) -> dict[str, Any]:
    from d4rt_agent.sam_orientany_d4rt_test.traj3d_orient import load_model, mirror_test, orient_target

    pending = [t for t in targets if force or not _done(t, run_dir, "orient")]
    if not pending:
        print("[orient] all targets already done, skipping model load", flush=True)
        return {}

    model = load_model()

    # The handedness check runs before any subject is oriented. If it fails, the
    # basis in traj3d_geometry is mirrored and every triad would point the wrong
    # way -- a failure that looks completely plausible on a plot.
    verdict = mirror_test(model)
    write_json(run_dir / "orientation_mirror_test.json", verdict)
    print(f"[orient] mirror test: {verdict['verdict']}", flush=True)

    results: dict[str, Any] = {}
    for target in pending:
        with _isolated("orient", target.slug):
            record = orient_target(target, run_dir, model)
            results[target.slug] = record
            print(
                f"[orient] {target.slug}: {record['num_ok']}/32 oriented, "
                f"{record['num_ambiguous']} ambiguous",
                flush=True,
            )
    _all_failed("orient", [t.slug for t in pending], results)
    return results


def run_stage_analyze(targets: list[Target], run_dir: Path, force: bool) -> dict[str, Any]:
    from d4rt_agent.sam_orientany_d4rt_test.traj3d_analyze import analyze_target

    results: dict[str, Any] = {}
    attempted: list[str] = []
    for target in targets:
        if not force and _done(target, run_dir, "analyze"):
            print(f"[analyze] {target.slug}: already done, skipping", flush=True)
            continue
        attempted.append(target.slug)
        with _isolated("analyze", target.slug):
            summary = analyze_target(target, run_dir)
            results[target.slug] = summary
            print(
                f"[analyze] {target.slug}: {summary['tracks_kept']}/{summary['tracks_total']} "
                f"tracks, camera rmse {summary['camera']['median_relative_rmse']:.4f}",
                flush=True,
            )
            for warning in summary["warnings"]:
                print(f"[analyze] {target.slug}: WARNING {warning}", flush=True)
    _all_failed("analyze", attempted, results)
    return results


RUNNERS = {
    "segment": run_stage_segment,
    "query": run_stage_query,
    "orient": run_stage_orient,
    "analyze": run_stage_analyze,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--only", nargs="*", default=None, help="slugs or relative paths")
    parser.add_argument(
        "--stages", nargs="*", default=list(STAGES), choices=list(STAGES),
        help="subset of stages to run, in the given order",
    )
    parser.add_argument("--force", action="store_true", help="redo completed stages")
    args = parser.parse_args()

    run_dir = args.run_dir or default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    targets = _selected(args.only)

    metadata = run_metadata(run_dir)
    metadata["stages_requested"] = list(args.stages)
    metadata["targets_selected"] = [t.slug for t in targets]
    write_json(run_dir / "run_metadata.json", metadata)

    print(f"run_dir={run_dir}", flush=True)
    print(f"targets={[t.slug for t in targets]}", flush=True)

    failures: dict[str, str] = {}
    for stage in args.stages:
        started = time.time()
        print(f"\n=== stage: {stage} ===", flush=True)
        try:
            RUNNERS[stage](targets, run_dir, args.force)
        except Exception:
            failures[stage] = traceback.format_exc()
            print(f"[{stage}] FAILED:\n{failures[stage]}", flush=True)
            # Later stages consume this one's output, so continuing would only
            # produce a confusing second failure.
            break
        print(f"=== stage {stage} took {time.time() - started:.1f}s ===", flush=True)

    write_json(
        run_dir / "run_status.json",
        {
            "completed_stages": [s for s in args.stages if s not in failures],
            "failures": failures,
            "target_failures": TARGET_ERRORS,
            "run_dir": str(run_dir),
        },
    )
    if failures:
        raise SystemExit(1)
    if TARGET_ERRORS:
        print(
            f"\nAll stages complete with {len(TARGET_ERRORS)} dropped target(s): "
            f"{sorted(TARGET_ERRORS)}",
            flush=True,
        )
    else:
        print(f"\nAll stages complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
