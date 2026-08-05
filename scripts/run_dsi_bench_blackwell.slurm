#!/usr/bin/env bash
# DSI-Bench evaluation of the simple_v2 agent, on the Blackwell (RTX PRO 6000) nodes.
# Derived from run_simple_v2_blackwell.slurm; same cu128 venv and offline HF setup.
#
#   sbatch --export=ALL,POINT_MODE=ensemble5 scripts/run_dsi_bench_blackwell.slurm
#   sbatch --export=ALL,POINT_MODE=ensemble5,LIMIT=2 scripts/run_dsi_bench_blackwell.slurm
#   sbatch --export=ALL,POINT_MODE=ensemble5,MODE=baseline scripts/run_dsi_bench_blackwell.slurm
#   sbatch --export=ALL,POINT_MODE=ensemble5,RESULTS_DIR=/scratch/run,MANIFEST_PATH=/scratch/run/manifest.json,ONLY_FILE=/scratch/question_ids.txt,FORCE=0 scripts/run_dsi_bench_blackwell.slurm
#
# The runner writes one file per question and skips questions already answered, so
# a job that runs out of time is continued by resubmitting the same command. That
# is preferable to sharding: each shard would reload the 14 GB D4RT checkpoint,
# which is the cost the single-load design exists to avoid.
#
# ---------------------------------------------------------------------------
# The full 1769-question census instead of the 25-question sample. RUN the file,
# do not sbatch it -- it submits itself with the allocation the census needs:
#
#   POINT_MODE=ensemble5 ./scripts/run_dsi_bench_blackwell.slurm --all-questions
#   POINT_MODE=ensemble5 ALL_QUESTIONS=1 ./scripts/run_dsi_bench_blackwell.slurm
#
# Executed that way, --all-questions picks up three things automatically:
#
#   partition cuda13pr.24h and --time=24:00:00, instead of the 3 h header below
#   results dir d4rt_agent/results/dsi_bench_full, the `_full` suffix appended
#   job name d4rt-dsi-full, logging to dsi_bench_full_%j.out
#
# The suffix is not cosmetic: the manifest lives at <results-dir>/manifest.json,
# so a shared directory would mean each design overwriting the other's selection
# and silently breaking the pairing with every previous run.
#
# `sbatch <this file>` still works and still honours --all-questions, but sbatch
# has already read the 3 h header by then, so only the results dir moves. Submit
# by executing the file when you want the allocation to follow the flag.
#
# Overridable per submit: SUBMIT_PARTITION, SUBMIT_TIME, SUBMIT_JOB_NAME,
# SUBMIT_OUTPUT.
#
# Build the census manifest once, up front, on the login node (CPU, seconds):
#
#   "$D4RT_PYTHON" -m d4rt_agent.dsi_bench_run --all-questions --dry-run \
#       --results-dir d4rt_agent/results/dsi_bench_full
#
# The census is ~45-50 GPU-hours for the agent pass, so it outlasts even a 24 h
# job. Resubmit the identical command to continue: the runner writes one file per
# question and skips every question already answered, so a resubmission picks up
# exactly the questions still missing. Both this script and the runner print the
# pending count before any GPU work starts.

#SBATCH --job-name=d4rt-dsi-bw
#SBATCH --partition=cuda13pr.4h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
# The site submit filter flattens the typed GPU request and also adds gpupr.4h,
# whose A100 nodes share the cuda13/gpupr features -- so pin to the Blackwell
# nodes explicitly. EPYC_9654 is unique to eu-g7-[001-010] among reachable GPU nodes.
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=8G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/Open-d4rt/logs/d4rt_agent/dsi_bench_%j.out

set -euo pipefail

# Resolved before anything cds, and used only on the submit-time path below --
# inside a job SLURM runs a spool copy, so this would not point at the repo.
SELF="$(readlink -f "${BASH_SOURCE[0]}")"

REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
D4RT_PYTHON=/cluster/work/igp_psr/spanwar/envs/d4rt_bw/bin/python
DEPS_DIR="$REPO_ROOT/.cache/qwen3vl_python"
LOCAL_DEPS="${TMPDIR:-/tmp/$USER}/qwen3vl_python"
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
MODEL_CACHE="$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"

# Accepted as a positional argument as well as an env var. The runner spells it
# as a flag, so repeating that spelling here keeps the submit line and the
# command it ends up running readable as the same thing.
ALL_QUESTIONS="${ALL_QUESTIONS:-0}"
for arg in "$@"; do
  case "$arg" in
    --all-questions) ALL_QUESTIONS=1 ;;
    *)
      echo "Unsupported argument: $arg (only --all-questions is accepted)" >&2
      exit 2
      ;;
  esac
done
if [[ "$ALL_QUESTIONS" != "0" && "$ALL_QUESTIONS" != "1" ]]; then
  echo "Unsupported ALL_QUESTIONS=$ALL_QUESTIONS (expected 0 or 1)" >&2
  exit 2
fi

RESULTS_DIR="${RESULTS_DIR:-d4rt_agent/results/dsi_bench}"
RESULTS_DIR="${RESULTS_DIR%/}"
# The census gets its own directory. The manifest lives inside the results dir,
# so sharing one would mean each design overwriting the other's selection --
# and a 1769-question manifest where 25 answers already sit looks like a run
# that is 1.4% done rather than a different experiment. Appending the suffix
# here is what makes --all-questions safe to hand to the ordinary launcher.
# Guarded against doubling, so an explicit RESULTS_DIR=.../dsi_bench_full is
# left as written instead of becoming dsi_bench_full_full.
if [[ "$ALL_QUESTIONS" == "1" && "$RESULTS_DIR" != *_full ]]; then
  RESULTS_DIR="${RESULTS_DIR}_full"
fi
# Resolved because the manifest default is built from it and RESULTS_DIR may be
# given as an absolute scratch path; the script otherwise runs from REPO_ROOT.
case "$RESULTS_DIR" in
  /*) RESULTS_DIR_ABS="$RESULTS_DIR" ;;
  *) RESULTS_DIR_ABS="$REPO_ROOT/$RESULTS_DIR" ;;
esac

# The sample pins its manifest to the canonical directory even when the answers
# go elsewhere, so a scratch run still asks the same 25 questions and stays
# comparable. A census has no canonical copy to pin to -- it is defined by the
# split, not by a draw -- so it takes the manifest sitting beside its answers.
if [[ "$ALL_QUESTIONS" == "1" ]]; then
  MANIFEST_PATH="${MANIFEST_PATH:-$RESULTS_DIR_ABS/manifest.json}"
  EXPECTED_DESIGN=census
else
  MANIFEST_PATH="${MANIFEST_PATH:-$REPO_ROOT/d4rt_agent/results/dsi_bench/manifest.json}"
  EXPECTED_DESIGN=latin_rectangle
fi
ONLY_FILE="${ONLY_FILE:-}"
FORCE="${FORCE:-0}"
MODE="${MODE:-agent}"
POINT_MODE="${POINT_MODE:?submit with POINT_MODE=centroid or POINT_MODE=ensemble5}"

if [[ "$POINT_MODE" != "centroid" && "$POINT_MODE" != "ensemble5" ]]; then
  echo "Unsupported POINT_MODE=$POINT_MODE" >&2
  exit 2
fi
if [[ "$MODE" != "agent" && "$MODE" != "baseline" ]]; then
  echo "Unsupported MODE=$MODE (expected agent or baseline)" >&2
  exit 2
fi
if [[ "$MODE" == "agent" && "$POINT_MODE" != "ensemble5" ]]; then
  echo "Phase 2 agent evaluation requires host-fixed POINT_MODE=ensemble5" >&2
  exit 2
fi
if [[ "$FORCE" != "0" && "$FORCE" != "1" ]]; then
  echo "Unsupported FORCE=$FORCE (expected 0 or 1)" >&2
  exit 2
fi

cd "$REPO_ROOT"
# Built once, up front, so a job never spends its allocation deciding which
# questions to ask -- and so a mistyped results dir fails here rather than
# quietly starting a different run.
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Manifest does not exist: $MANIFEST_PATH" >&2
  if [[ "$ALL_QUESTIONS" == "1" ]]; then
    echo "Build it first, on the login node:" >&2
    echo "  $D4RT_PYTHON -m d4rt_agent.dsi_bench_run --all-questions --dry-run \\" >&2
    echo "      --results-dir $RESULTS_DIR" >&2
  fi
  exit 2
fi

# Checked in both directions. dsi_bench_run already refuses --all-questions
# against a sample manifest, but not the reverse -- and the reverse is the one
# that stays silent: answering the first 25 of a census and filing them where
# the sample's results are read from produces a plausible-looking directory
# that no longer means what its consumers assume.
DESIGN="$("$D4RT_PYTHON" -c "import json,sys;print(json.load(open(sys.argv[1]))['sampling']['design'])" "$MANIFEST_PATH")"
if [[ "$DESIGN" != "$EXPECTED_DESIGN" ]]; then
  echo "Refusing to run: $MANIFEST_PATH has design=$DESIGN, expected $EXPECTED_DESIGN." >&2
  if [[ "$ALL_QUESTIONS" == "1" ]]; then
    echo "That is the 25-question sample, not the full split." >&2
  else
    echo "That is the full split, not the 25-question sample; pass --all-questions." >&2
  fi
  exit 2
fi

TOTAL="$("$D4RT_PYTHON" -c "import json,sys;print(len(json.load(open(sys.argv[1]))['questions']))" "$MANIFEST_PATH")"
# Spelled out rather than piped into `wc -l`: `find` on a missing directory
# exits non-zero, which under `set -euo pipefail` kills the job before it prints
# anything, and the directory is legitimately missing on the first submission.
DONE_ALREADY=0
if [[ -d "$RESULTS_DIR/answers" ]]; then
  DONE_ALREADY="$(find "$RESULTS_DIR/answers" -maxdepth 1 -name '*.json' | wc -l)"
fi

ONLY_IDS=()
if [[ -n "$ONLY_FILE" ]]; then
  if [[ ! -f "$ONLY_FILE" ]]; then
    echo "ONLY_FILE does not exist: $ONLY_FILE" >&2
    exit 2
  fi
  declare -A SEEN_ONLY_IDS=()
  while IFS= read -r raw_id || [[ -n "$raw_id" ]]; do
    question_id="${raw_id#"${raw_id%%[![:space:]]*}"}"
    question_id="${question_id%"${question_id##*[![:space:]]}"}"
    if [[ -z "$question_id" ]]; then
      continue
    fi
    if [[ -n "${SEEN_ONLY_IDS[$question_id]:-}" ]]; then
      echo "Duplicate question ID in ONLY_FILE: $question_id" >&2
      exit 2
    fi
    SEEN_ONLY_IDS["$question_id"]=1
    ONLY_IDS+=("$question_id")
  done < "$ONLY_FILE"
  if [[ "${#ONLY_IDS[@]}" -eq 0 ]]; then
    echo "ONLY_FILE contains no non-empty question IDs: $ONLY_FILE" >&2
    exit 2
  fi
fi

echo "ALL_QUESTIONS=$ALL_QUESTIONS"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "MANIFEST_PATH=$MANIFEST_PATH (design=$DESIGN, $TOTAL questions)"
echo "ANSWERS ALREADY ON DISK=$DONE_ALREADY"
echo "ONLY_FILE=${ONLY_FILE:-<unset>} (${#ONLY_IDS[@]} selected ID(s))"
echo "FORCE=$FORCE"
echo "MODE=$MODE  immutable POINT_MODE=$POINT_MODE  LIMIT=${LIMIT:-all}"
echo "GROUNDING_TOOL=isolated_qwen bbox_tokens=96 points_tokens=192 seed=42"
echo "CODE_SHA=$(git rev-parse HEAD)"

# ---------------------------------------------------------------------------
# Submit-time re-exec.
#
# sbatch reads the #SBATCH block out of this file before the body ever runs, so
# the header cannot depend on --all-questions; there is no in-file way to make
# it conditional. Executing this file instead of submitting it lets the
# allocation be chosen here and passed on the sbatch command line, where it
# outranks the header:
#
#   POINT_MODE=ensemble5 ./scripts/run_dsi_bench_blackwell.slurm --all-questions
#
# Everything above this point -- argument parsing, POINT_MODE/MODE validation,
# and the manifest existence and design checks -- has therefore already run on
# the login node, so a bad submit fails now rather than after queueing.
#
# Inside a job SLURM_JOB_ID is set and this is skipped, so `sbatch <this file>`
# still behaves exactly as it did before, with the 3 h header.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "$ALL_QUESTIONS" == "1" ]]; then
    SUBMIT_PARTITION="${SUBMIT_PARTITION:-cuda13pr.24h}"
    SUBMIT_TIME="${SUBMIT_TIME:-24:00:00}"
    SUBMIT_JOB_NAME="${SUBMIT_JOB_NAME:-d4rt-dsi-full}"
    SUBMIT_OUTPUT="${SUBMIT_OUTPUT:-$REPO_ROOT/logs/d4rt_agent/dsi_bench_full_%j.out}"
  else
    SUBMIT_PARTITION="${SUBMIT_PARTITION:-cuda13pr.4h}"
    SUBMIT_TIME="${SUBMIT_TIME:-03:00:00}"
    SUBMIT_JOB_NAME="${SUBMIT_JOB_NAME:-d4rt-dsi-bw}"
    SUBMIT_OUTPUT="${SUBMIT_OUTPUT:-$REPO_ROOT/logs/d4rt_agent/dsi_bench_%j.out}"
  fi

  # Re-stated on the command line rather than relied on through --export=ALL:
  # ALL_QUESTIONS=1 given as a plain shell assignment is not in the environment,
  # and the resubmitted job would then quietly run the sample.
  SUBMIT_ARGS=()
  if [[ "$ALL_QUESTIONS" == "1" ]]; then
    SUBMIT_ARGS+=(--all-questions)
  fi

  mkdir -p "$REPO_ROOT/logs/d4rt_agent"
  echo "Submitting from login node:"
  echo "  partition=$SUBMIT_PARTITION time=$SUBMIT_TIME name=$SUBMIT_JOB_NAME"
  echo "  $TOTAL questions, $DONE_ALREADY already answered, $((TOTAL - DONE_ALREADY)) pending"
  exec sbatch \
    --partition="$SUBMIT_PARTITION" \
    --time="$SUBMIT_TIME" \
    --job-name="$SUBMIT_JOB_NAME" \
    --output="$SUBMIT_OUTPUT" \
    --export=ALL \
    "$SELF" "${SUBMIT_ARGS[@]}"
fi

mkdir -p logs/d4rt_agent "$RESULTS_DIR" "$LOCAL_DEPS"
cp -a "$DEPS_DIR/." "$LOCAL_DEPS/"

export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DATASETS_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$LOCAL_DEPS${PYTHONPATH:+:$PYTHONPATH}"

MODEL_DIR="$(find "$MODEL_CACHE" -mindepth 1 -maxdepth 1 -type d -print -quit)"
if [[ -z "$MODEL_DIR" ]]; then
  echo "Offline Qwen snapshot missing. Run scripts/prepare_qwen3vl_login.sh first." >&2
  exit 2
fi
export MODEL_DIR

EXTRA=()
if [[ "$ALL_QUESTIONS" == "1" ]]; then
  EXTRA+=(--all-questions)
fi
if [[ "$MODE" == "baseline" ]]; then
  EXTRA+=(--baseline)
fi
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA+=(--limit "$LIMIT")
fi
if [[ "${#ONLY_IDS[@]}" -gt 0 ]]; then
  EXTRA+=(--only "${ONLY_IDS[@]}")
fi
if [[ "$FORCE" == "1" ]]; then
  EXTRA+=(--force)
fi

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

"$D4RT_PYTHON" -m d4rt_agent.dsi_bench_run \
  --results-dir "$RESULTS_DIR" \
  --manifest "$MANIFEST_PATH" \
  --point-mode "$POINT_MODE" \
  --qwen-model "$MODEL_DIR" \
  --d4rt-config checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-checkpoint checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --d4rt-dtype bfloat16 \
  --device cuda \
  --seed 42 \
  --max-new-tokens 768 \
  --max-steps 14 \
  "${EXTRA[@]}"
