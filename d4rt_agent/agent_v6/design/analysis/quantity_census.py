"""What the three benchmarks ask, grouped by the physical quantity that decides the answer.

Maps every existing per-question kind label (DSI category, Dyn decomposition census, VLM4D
exploration manifest) to one V6 quantity and the V6 tool that measures it, and attaches the
evidence on whether measuring helped:
  Dyn  - D7 study (2,800 eval questions): frames-only, option log-probs, answer-with-measurement (over questions that have one);
         V5 legacy Point4D on the same questions, keyed by the study's rule kind.
  VLM4D - V5 legacy Point4D on all 1,816 questions, keyed by the manifest kind.
Reads saved JSON only. Prints a markdown table.
"""
import collections
import csv
import json
from pathlib import Path

REPO = Path("/cluster/work/igp_psr/spanwar/shak-spatial-agent")
RES = REPO / "d4rt_agent/results"

# quantity id -> (description, V6 tool)
QUANTITIES = {
    "camera_motion": ("Camera motion: path, translation, turning, orbit", "track_camera"),
    "camera_tempo": ("Camera tempo and steadiness: speeding up, shake", "track_camera"),
    "zoom": ("Lens zoom", "track_camera_zoom"),
    "revisit": ("Whether the end view returns to the start view", "check_scene_revisit"),
    "entity_motion": ("An entity's motion and direction", "track_entity"),
    "entity_speed": ("An entity's speed profile, incl. metric speed", "track_entity (speed section)"),
    "entity_camera": ("Entity and camera: distance, height, which side is filmed", "track_entity (camera / subject)"),
    "entity_entity": ("Two entities or an entity and a landmark: distance, relative position", "track_entity (others)"),
    "rotation": ("Rotation sense, turning, facing", "track_orientation"),
    "on_screen": ("On-screen size and visibility", "get_segmentation_mask"),
    "count": ("Counting instances, incl. with a motion predicate", "detect_all_entities + track_multiple_entities; VLM counts"),
    "visual": ("Events, actions, repetitions, hands, appearance", "ask-video tool / VLM"),
    "unlabelled": ("No label", "-"),
}

DYN = {
    "camera_path": "camera_motion", "camera_translation": "camera_motion", "orbit_direction": "camera_motion",
    "camera_rhythm": "camera_tempo", "camera_shake": "camera_tempo", "camera_zoom": "zoom",
    "scene_revisit": "revisit", "subject_direction": "entity_motion", "subject_motion_profile": "entity_speed",
    "metric_speed": "entity_speed", "cam_subject_range": "entity_camera", "cam_subject_elevation": "entity_camera",
    "side_filmed": "entity_camera", "pair_relation": "entity_entity", "landmark_relation": "entity_entity",
    "metric_distance": "entity_entity", "subject_visibility": "on_screen", "apparent_size": "on_screen",
    "instance_count": "count", "tracked_objects_count": "count", "event_order": "visual", "semantic": "visual",
    "unread": "unlabelled", "unparsed": "unlabelled", "other": "unlabelled",
}
VLM4D = {
    "subject_direction": "entity_motion", "traj_subject_own_frame": "entity_motion", "camera_path": "camera_motion",
    "camera_translation": "camera_motion", "orbit_direction": "camera_motion", "cam_subject_range": "entity_camera",
    "camera_viewpoint": "entity_camera", "relative_position": "entity_entity", "landmark_relation": "entity_entity",
    "pair_relation": "entity_entity", "rotation_sense": "rotation", "subject_turn": "rotation",
    "count_with_predicate": "count", "repetition_count": "visual", "hand_attribution": "visual",
    "event_order": "visual", "semantic": "visual", "other": "unlabelled",
}
DSI = {0: "entity_motion", 1: "entity_motion", 2: "camera_motion", 3: "camera_motion", 4: "entity_camera",
       5: "entity_camera"}


def answers(folder):
    out = {}
    for f in (RES / folder / "answers").glob("*.json"):
        d = json.load(open(f))
        out[d["question_id"]] = bool(d.get("correct"))
    return out


def main():
    counts = {b: collections.Counter() for b in ("DSI", "Dyn", "VLM4D")}
    with open("/cluster/work/igp_psr/spanwar/datasets/DSI-Bench/metadatas/std.csv") as fh:
        for row in csv.DictReader(fh):
            counts["DSI"][DSI[int(row["cate"])]] += 1
    census = json.load(open(REPO / "dyn_dataset_exploration/outputs/decomposition_census/census.json"))["counts"]
    for kind, n in census.items():
        counts["Dyn"][DYN[kind]] += n
    manifest = json.load(open(REPO / "vlm4d_dataset_exploration/logs/manifest.json"))["questions"]
    for q in manifest:
        counts["VLM4D"][VLM4D[q["kind"]]] += 1

    # Dyn evidence: D7 study rows (rule kind) joined with V5 legacy Point4D answers
    rows = json.load(open(REPO / "dyn_dataset_exploration/outputs/decomposition_eval/eval_rows.json"))
    v5_dyn = answers("agent_v5_eval2800_point4d_20261005T183754Z")
    dyn_ev = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])  # n, frames, letters, evidence, v5, n_evidence
    for r in rows:
        e = dyn_ev[DYN.get(r["kind"], "unlabelled")]
        e[0] += 1
        e[1] += r["frames"] == r["gt"]
        e[2] += r["letters"] == r["gt"]
        e[3] += r.get("evidence") == r["gt"]
        e[5] += r.get("evidence") is not None
        e[4] += v5_dyn.get(r["qid"], False)
    v5_vlm = answers("agent_v5_vlm4d_point4d_20261010T012544Z")
    vlm_ev = collections.defaultdict(lambda: [0, 0])
    for q in manifest:
        e = vlm_ev[VLM4D[q["kind"]]]
        e[0] += 1
        e[1] += v5_vlm.get(q["qid"], False)

    pct = lambda a, n: f"{100 * a / n:.0f}" if n else "—"
    print("| Quantity | V6 tool | DSI | Dyn | VLM4D | All | Dyn eval n | frames | letters | with measurement (coverage) | V5 Dyn | V5 VLM4D |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for qid, (desc, tool) in QUANTITIES.items():
        d, y, v = counts["DSI"][qid], counts["Dyn"][qid], counts["VLM4D"][qid]
        n, f, l, ev, v5, ne = dyn_ev[qid]
        vn, vv = vlm_ev[qid]
        print(f"| {desc} | `{tool}` | {d} | {y} | {v} | {d + y + v} | {n} | {pct(f, n)} | {pct(l, n)} | {pct(ev, ne)} ({ne}) | "
              f"{pct(v5, n)} | {pct(vv, vn)} |")
    tot = {b: sum(c.values()) for b, c in counts.items()}
    print(f"\nTotals: DSI {tot['DSI']}, Dyn {tot['Dyn']}, VLM4D {tot['VLM4D']}, all {sum(tot.values())}")


if __name__ == "__main__":
    main()
