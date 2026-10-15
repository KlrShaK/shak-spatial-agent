"""Build the lightweight, executable research notebook from saved experiments."""

import argparse
import json
from pathlib import Path
import textwrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cells = []

    def markdown(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"})

    def code(text):
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                      "source": textwrap.dedent(text).strip() + "\n"})

    markdown("""
    # AgentV5: what must a measurement distinguish?

    Algorithm design informed by **DSI-Bench, Dyn-Bench and VLM4D together**.
    No model training, threshold fitting, or train-on-two/test-on-third protocol.
    This notebook reads the full corpus audit, recorded V5 traces, exact synthetic
    counterexamples, and a five-clip cached-geometry comparison. It performs no
    model inference and does not select benchmark answers.

    Read `FINDINGS.md` beside this notebook for interpretation and design decisions.
    Lexical screens retrieve examples; they are not validated coverage labels.
    The casebook reviews language, not video ground truth. Synthetic failures and
    disagreements between estimated tracks do not measure benchmark error rates.
    """)
    code("""
    import json
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython.display import display, Image

    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "d4rt_agent").is_dir())
    RESEARCH = ROOT / "d4rt_agent/agent_v5/research"
    OUT = ROOT / "d4rt_agent/results/agent_v5_capability_research_20261013"
    def read(name):
        return json.loads((OUT / name).read_text())
    corpus = read("corpus_blind.json")
    traces = read("trace_rows.json")
    counts = read("census.json")
    counter = read("counterexamples.json")
    pd.set_option("display.max_colwidth", 120)
    assert len(corpus) == 10585 and len(traces) == 5416
    assert not any("answer" in r or "gold" in r for r in corpus)
    display(pd.DataFrame({b: {k: v for k, v in d.items() if k in ("questions", "clips")}
                          for b, d in counts.items()}).T)
    """)
    markdown("""
    ## Which distinctions appear only in the options?

    Each screen is independently applied to the question and option text. Flags
    overlap and include contextual mentions. The narrow `absence` screen misses
    many short phrasings such as "no bus there". The separate existing text-only
    absence parser identifies 795 VLM4D questions with an absence option.
    """)
    code("""
    screen_table = pd.DataFrame([
        {"screen": flag, **{b: d["screens"][flag]["either"] for b, d in counts.items()},
         **{b + " options only": d["screens"][flag]["options_only"] for b, d in counts.items()}}
        for flag in counts["DSI"]["screens"]
    ]).set_index("screen")
    display(screen_table)
    from vlm4d_dataset_exploration.taxonomy import is_absent
    print("VLM4D presence/absence alternatives:", sum(any(is_absent(v) for v in r["options"].values())
                                                   for r in corpus if r["benchmark"] == "VLM4D"))
    """)
    markdown("""
    ## What did the recorded agent actually attempt?

    Accuracy is descriptive; these runs have different coverage. A missing
    `available=False` flag does not prove that the tool measured the right thing.
    Frames below count calls, not distinct questions. Compare kinds and frames
    with the casebook's requirements before attributing an error to perception.
    """)
    code("""
    summary = read("trace_summary.json")
    overall = pd.DataFrame([{ "benchmark": b, **{k: v for k, v in s["overall"].items()
                                               if k not in ("kinds", "track_frames")}}
                            for b, s in summary.items()]).set_index("benchmark")
    overall["accuracy"] = overall["correct"] / overall["n"]
    display(overall)
    display(pd.DataFrame({b: s["overall"]["track_frames"] for b, s in summary.items()}).fillna(0).astype(int))
    corpus_by_id = {r["qid"]: r for r in corpus}
    own_subject = [r for r in traces if r["benchmark"] == "DSI" and corpus_by_id[r["qid"]]["group"] in ("0", "1")]
    print("DSI own-frame subject questions:", len(own_subject),
          "attempted own-frame:", sum("own" in r["track_frames"] for r in own_subject),
          "scene-frame only:", sum(bool(r["track_frames"]) and set(r["track_frames"]) == {"scene"} for r in own_subject))
    kinds = pd.DataFrame(traces).groupby(["benchmark", "kind"]).agg(n=("correct", "size"), accuracy=("correct", "mean"))
    display(kinds)
    """)
    markdown("""
    ## Exact counterexamples against current code

    These use production measurement functions with synthetic inputs. The
    experiment does not change thresholds. Twelve reproduced failures include
    information loss, a misleading report, unsupported schema predicates, and
    wrong time/identity semantics; they are not twelve independent accuracy gains.
    """)
    code("""
    display(pd.DataFrame([{k: r[k] for k in ("case", "reproduced", "production_function", "invariant")}
                          for r in counter["cases"]]))
    series = counter["series"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(series["time"], series["return_path_x"])
    axes[0].set(title="Moving camera classified static", xlabel="Time (s)", ylabel="Position (synthetic units)")
    axes[1].plot(series["irregular_measured_speed"], label="V5 mean-FPS calculation")
    axes[1].axhline(1, color="black", linestyle="--", label="Exact constant speed")
    axes[1].set(title="Resampling changes reported speed", xlabel="Measurement index", ylabel="Speed (units/s)")
    axes[1].legend(fontsize=8)
    axes[2].plot(series["surface_x"], label="Visible-surface median")
    axes[2].set(title="Stationary surface, changing occlusion", xlabel="Frame", ylabel="Inferred x (synthetic units)")
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_diagnostics.png", dpi=160)
    plt.show()
    """)
    markdown("""
    ## Review the evidence requirements of actual questions

    Change `selected_id` below to inspect any of the 32 curated cases. These
    annotations preserve uncertainty about frame/turn semantics. They do not
    infer the answer or treat previous rule labels as truth.
    """)
    code("""
    annotations = json.loads((RESEARCH / "casebook.json").read_text())["cases"]
    by_id = {r["qid"]: r for r in corpus}
    trace_by_id = {r["qid"]: r for r in traces}
    assert len(annotations) == 32 and all(r["qid"] in by_id for r in annotations)
    reviewed = [{**by_id[a["qid"]], **a, "recorded_kind": trace_by_id.get(a["qid"], {}).get("kind")}
                for a in annotations]
    display(pd.DataFrame(reviewed)[["benchmark", "question", "needs", "recorded_kind", "finding"]])
    selected_id = "vlm4d_real_davis-baseball_3"
    display(by_id[selected_id])
    display(next(a for a in annotations if a["qid"] == selected_id))
    """)
    markdown("""
    ## Cached tracks: a diagnostic, not a replacement algorithm

    Fixed-seed Point4D displacement and visible-surface medians use different
    correspondences. Neither is ground truth. The seed comparison uses the
    existing frame-0 query grid without a fitted confidence threshold. Mask
    membership is an agreement check; projected points inside a mask can still
    have incorrect depth, and correct hidden points can lie outside the visible mask.
    """)
    code("""
    cached = read("cached_tracks.json")["cases"]
    fields = ["clip", "status", "instances", "seed_points", "mask_membership_fraction", "net_disagreement_scales"]
    display(pd.DataFrame(cached).reindex(columns=fields))
    for r in cached:
        plot = OUT / ("tracks_" + r["clip"].strip("_") + ".png")
        if plot.exists():
            display(Image(filename=str(plot)))
    """)
    markdown("""
    ## Next controlled comparisons

    Use all three datasets to select diagnostic cases and guide the shared
    algorithm. Separate fixes to frame/reduction semantics from new primitives,
    routing changes, and reporting changes. Keep the model as the final answer
    chooser. Compare matched questions, source clips, backbones and visual inputs;
    include DSI reverse/flip augmentations and report each benchmark separately.

    Reproduce the underlying CPU jobs with commands in `README.md`. This notebook
    intentionally does not submit jobs or load perception models automatically.
    """)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 4}
    path = Path(__file__).with_name("capability_audit.ipynb")
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    if args.execute:
        import nbformat
        from nbclient import NotebookClient

        book = nbformat.read(path, as_version=4)
        NotebookClient(book, timeout=240, kernel_name="python3").execute(cwd=str(Path(__file__).resolve().parents[3]))
        nbformat.write(book, path)
    print(path)


if __name__ == "__main__":
    main()
