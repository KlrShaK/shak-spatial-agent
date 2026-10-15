"""Freeze source for reproducible SLURM runs without copying models, media or result caches."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def snapshot(root, destination, baseline=False):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("use a new snapshot directory; frozen sources are never overwritten")
    names = subprocess.check_output(["rg", "--files", "d4rt_agent", "dyn_dataset_exploration", "dsi_dataset_exploration", "vlm4d_dataset_exploration",
                                     "scripts", "-g", "*.py", "-g", "*.md", "-g", "*.json", "-g", "*.sh",
                                     "-g", "*.slurm", "-g", "*.yaml", "-g", "*.yml"], cwd=root, text=True).splitlines()
    destination.mkdir(parents=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    restored = []
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
        if baseline and name.startswith("d4rt_agent/agent_v5/") and name.endswith(".py") and not name.endswith("/run.py"):
            original = subprocess.run(["git", "show", f"{revision}:{name}"], cwd=root, capture_output=True)
            if original.returncode == 0:
                target.write_bytes(original.stdout)
                restored.append(name)
        if baseline and name.startswith("d4rt_agent/agent_v5/prompts/"):
            original = subprocess.run(["git", "show", f"{revision}:{name}"], cwd=root, capture_output=True)
            if original.returncode == 0:
                target.write_bytes(original.stdout)
                restored.append(name)
    hashes = {name: hashlib.sha256((destination / name).read_bytes()).hexdigest() for name in names}
    manifest = {"git_revision": revision, "baseline": baseline, "restored_from_revision": restored,
                "runner_note": "Current dataset/scoring runner retained; baseline executes original legacy agent and measurements.",
                "files": hashes}
    (destination / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    manifest = snapshot(Path(__file__).resolve().parents[3], args.out, args.baseline)
    print(json.dumps({"out": str(args.out), "files": len(manifest["files"]), "baseline": args.baseline}))


if __name__ == "__main__":
    main()
