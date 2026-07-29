# Phase 0 — preserve the failed experiment, merge only the four grounding-probe commits, and create the implementation branch

## Purpose

This phase turns the current mixed repository state into three clearly separated histories:

1. `main` receives only the four already-reviewed isolated Qwen grounding-probe commits.
2. The failed forced-grounding prompt experiment, its 25-question output, and the planning drafts are preserved on a local archive branch.
3. All new orchestration and grounding-tool work starts from a clean feature branch based on the updated `main`.

This is repository preparation only. Do not implement Phase 1 or Phase 2 while following this file.

## Non-negotiable rules

- Work locally. Do not push any branch or commit.
- Do not use `git reset --hard`, `git checkout --`, `git clean`, or force-delete a branch.
- Do not use `git add -A` or `git add .`. Stage only the paths named below.
- Do not merge the archive branch into `main`.
- Do not merge the uncommitted forced-grounding prompt experiment into `main`.
- Do not rewrite the four existing grounding-probe commits.
- Stop immediately if any expected branch, hash, or file set differs from this document. Investigate the difference before continuing.
- A non-zero exit from a verification command is a stop condition unless the step explicitly says that non-zero is expected.

## Expected starting state

Run every command from:

```text
/cluster/work/igp_psr/spanwar/Open-d4rt
```

The expected branches and commits are:

| Reference | Expected commit | Meaning |
| --- | --- | --- |
| `main` | `3172b84f8ed266980ee01407fc16871e0e7fd517` | Base before the isolated grounding probe |
| `origin/main` | `3172b84f8ed266980ee01407fc16871e0e7fd517` | Same local reference; no fetch is required |
| `test/grounding` | `7e4108642f769a5db671e72a79338eabb30e83d7` | Tip containing exactly four desired commits |
| `origin/test/grounding` | `7e4108642f769a5db671e72a79338eabb30e83d7` | Same existing remote-tracking reference |

The four commits that must reach `main`, in oldest-to-newest order, are:

1. `833bf2e3a80eb1116de50dcf774d6e8cc9603864` — `feat(d4rt-agent): add isolated Qwen grounding probe`
2. `08e4076b4cba5a7b42c616fc2d772177b23d4140` — `feat(d4rt-agent): add grounding debug launchers`
3. `fffc2479f50e58ab6c12b87a2d435118d555e339` — `eval(d4rt-agent): record Qwen grounding probe results`
4. `7e4108642f769a5db671e72a79338eabb30e83d7` — `docs(d4rt-agent): add Qwen grounding debug analysis`

The current working tree is expected to contain the failed forced-grounding experiment:

- modified implementation and prompt files:
  - `d4rt_agent/dsi_bench_report.py`
  - `d4rt_agent/prompts/dsi_bench_system.md`
  - `d4rt_agent/test_dsi_bench.py`
  - `scripts/run_dsi_bench_a100.slurm`
  - `scripts/run_dsi_bench_blackwell.slurm`
- modified failed-run outputs:
  - `d4rt_agent/results/dsi_bench/DSI_BENCH_RESULTS.md`
  - `d4rt_agent/results/dsi_bench/FINDINGS.md`
  - all 25 tracked files under `d4rt_agent/results/dsi_bench/answers/`
  - the modified files under `d4rt_agent/results/dsi_bench/groundings/`
- untracked failed-run/planning material:
  - `d4rt_agent/results/dsi_bench/DSI_BENCH_RESULTS_INTERMEDIATE.md`
  - `d4rt_agent/plans/phase_1_draft.md`
  - `d4rt_agent/plans/phase_2_draft.md`
  - the finalized `phase_0.md`, `phase_1.md`, and `phase_2.md` files

The exact number of modified grounding JPEGs may vary only if the report was regenerated again after this plan was written. Any other unexpected path must be reviewed before staging.

## Target branch layout

At the end of this phase:

```text
3172b84  original main
    |
    +-- 833bf2e -- 08e4076 -- fffc247 -- 7e41086
                                           |\
                                           | \-- archive/failed-forced-grounding-prompt
                                           |       \-- archive implementation commit
                                           |       \-- archive evaluation commit
                                           |       \-- archive planning commit
                                           |
                                           +-- main
                                           +-- test/grounding
                                           \-- feat/qwen-grounding-tool-orchestration
                                                   \-- finalized plan-files commit
```

The archive commit hashes are intentionally not predicted because they do not exist yet.

## Step 0.1 — preflight without changing anything

Run:

```bash
pwd
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git rev-parse main
git rev-parse origin/main
git rev-parse test/grounding
git rev-parse origin/test/grounding
git status --short --branch
git log --format='%H %s' --reverse main..test/grounding
```

Expected results:

- Both directory commands identify `/cluster/work/igp_psr/spanwar/Open-d4rt`.
- The current branch is `test/grounding`.
- `HEAD`, `test/grounding`, and `origin/test/grounding` equal `7e410864...`.
- `main` and `origin/main` equal `3172b84f...`.
- The final command prints exactly the four commits listed above and no fifth commit.

Then prove the history is a fast-forward:

```bash
git merge-base --is-ancestor main test/grounding
```

Expected result: exit code 0 and no output.

### Stop conditions

Stop and report the discrepancy if:

- the current branch is not `test/grounding`;
- either named branch is missing;
- any expected hash differs;
- `main..test/grounding` contains more or fewer than four commits;
- the merge-base check fails;
- there are uncommitted paths unrelated to the failed forced-grounding run or these plans.

Do not “fix” an unexpected state with reset, checkout, clean, rebase, or force.

## Step 0.2 — inspect and record the dirty state

Run:

```bash
git diff --name-status
git ls-files --others --exclude-standard
git diff --stat
```

Save the terminal output in the implementation notes or task log. The goal is to have a before/after inventory.

Review the prompt diff and result headline before archiving:

```bash
git diff -- d4rt_agent/prompts/dsi_bench_system.md
git diff -- d4rt_agent/results/dsi_bench/FINDINGS.md
```

Confirm that the dirty experiment is the failed prompt-only forced-grounding run:

- 7/25 correct;
- 21/25 completed;
- four failures;
- the unchanged Qwen-only baseline remains 13/25;
- the previous committed D4RT agent remains 10/25.

If the dirty changes represent a different experiment, stop and rename/re-scope the archive branch before continuing.

## Step 0.3 — create the local archive branch

First ensure the intended archive name is unused:

```bash
git branch --list archive/failed-forced-grounding-prompt
```

Expected result: no output.

Create the branch without altering the working tree:

```bash
git switch -c archive/failed-forced-grounding-prompt
```

Verify:

```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

Expected:

- current branch is `archive/failed-forced-grounding-prompt`;
- `HEAD` is still `7e410864...`;
- all prior modifications and untracked files are still present.

## Step 0.4 — archive the failed experiment implementation

Stage only these files:

```bash
git add -- d4rt_agent/dsi_bench_report.py
git add -- d4rt_agent/prompts/dsi_bench_system.md
git add -- d4rt_agent/test_dsi_bench.py
git add -- scripts/run_dsi_bench_a100.slurm
git add -- scripts/run_dsi_bench_blackwell.slurm
```

Inspect the staged set:

```bash
git diff --cached --name-status
git diff --cached --stat
```

Expected staged paths: exactly the five paths above.

Commit:

```bash
git commit -m "experiment(d4rt-agent): archive forced-grounding prompt trial"
```

Verify:

```bash
git show --stat --oneline HEAD
git status --short
```

The implementation files should no longer appear as dirty. Results and plan files should remain dirty/untracked.

If an unintended file was staged before committing, unstage that specific path with:

```bash
git restore --staged -- path/to/unintended-file
```

Do not amend or rewrite a commit after it has been created unless the user explicitly requests history rewriting.

## Step 0.5 — archive the failed 25-question evaluation

Stage only the failed evaluation artifacts:

```bash
git add -- d4rt_agent/results/dsi_bench/DSI_BENCH_RESULTS.md
git add -- d4rt_agent/results/dsi_bench/FINDINGS.md
git add -- d4rt_agent/results/dsi_bench/DSI_BENCH_RESULTS_INTERMEDIATE.md
git add -- d4rt_agent/results/dsi_bench/answers
git add -- d4rt_agent/results/dsi_bench/groundings
```

Inspect before committing:

```bash
git diff --cached --name-status
git diff --cached --stat
```

Required checks:

- all 25 answer JSON files are staged;
- the failed final report and findings are staged;
- the intermediate report is staged;
- only DSI result artifacts are staged;
- no baseline JSON, manifest, video, GIF, or contact-sheet file was unintentionally added.

Commit:

```bash
git commit -m "eval(d4rt-agent): record failed forced-grounding DSI trial"
```

Verify:

```bash
git show --stat --oneline HEAD
git status --short
```

Only plan files should now remain untracked.

## Step 0.6 — archive the drafts and finalized plans

Stage exactly:

```bash
git add -- d4rt_agent/plans/phase_0.md
git add -- d4rt_agent/plans/phase_1.md
git add -- d4rt_agent/plans/phase_2.md
git add -- d4rt_agent/plans/phase_1_draft.md
git add -- d4rt_agent/plans/phase_2_draft.md
```

Inspect:

```bash
git diff --cached --name-status
```

Expected: five added Markdown files and no other path.

Commit:

```bash
git commit -m "docs(d4rt-agent): archive orchestration and grounding plans"
```

Verify the archive branch is clean:

```bash
git status --short --branch
git log --oneline --decorate -6
```

Expected:

- no short-status entries below the branch header;
- the latest three commits are the archive implementation, evaluation, and documentation commits;
- their parent chain reaches `7e410864...`.

Record the archive tip:

```bash
git rev-parse archive/failed-forced-grounding-prompt
```

This commit is the recovery point for every dirty file present at the start.

## Step 0.7 — fast-forward local `main` by exactly four commits

Switch to `main`:

```bash
git switch main
```

Verify before merging:

```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

Expected:

- branch is `main`;
- `HEAD` is `3172b84f...`;
- worktree is clean.

Fast-forward only:

```bash
git merge --ff-only test/grounding
```

Verify:

```bash
git rev-parse HEAD
git status --short --branch
git log --format='%H %s' --reverse 3172b84f8ed266980ee01407fc16871e0e7fd517..main
```

Expected:

- `main` now equals `7e410864...`;
- worktree is clean;
- the final command prints exactly the four approved commits, in the order listed in “Expected starting state.”

Prove that the archive-only commits did not enter `main`:

```bash
git log --oneline main..archive/failed-forced-grounding-prompt
git merge-base --is-ancestor archive/failed-forced-grounding-prompt main
```

Expected:

- the first command lists the three archive-only commits;
- the second command exits non-zero. That non-zero exit is expected because the archive tip must not be an ancestor of `main`.

Also verify the desired isolated-grounding files now exist on `main`:

```bash
git ls-tree -r --name-only main d4rt_agent/qwen_grounding_debug.py
git ls-tree -r --name-only main d4rt_agent/results/qwen_grounding_debug
git ls-tree -r --name-only main scripts/run_qwen_grounding_debug_blackwell.slurm
```

## Step 0.8 — create the Phase 1/2 feature branch

Ensure the feature-branch name is unused:

```bash
git branch --list feat/qwen-grounding-tool-orchestration
```

Expected: no output.

Create it from the updated `main`:

```bash
git switch -c feat/qwen-grounding-tool-orchestration
```

Verify:

```bash
git branch --show-current
git rev-parse HEAD
git rev-parse main
git status --short
```

Expected:

- current branch is `feat/qwen-grounding-tool-orchestration`;
- feature `HEAD` and `main` both equal `7e410864...`;
- worktree is clean.

## Step 0.9 — restore only the finalized plans onto the feature branch

The archive branch contains both drafts and final plans. Restore only the three final plans:

```bash
git restore --source=archive/failed-forced-grounding-prompt -- d4rt_agent/plans/phase_0.md
git restore --source=archive/failed-forced-grounding-prompt -- d4rt_agent/plans/phase_1.md
git restore --source=archive/failed-forced-grounding-prompt -- d4rt_agent/plans/phase_2.md
```

Do not restore:

- `phase_1_draft.md`;
- `phase_2_draft.md`;
- any failed prompt, code, result, or overlay.

Verify:

```bash
git status --short
git ls-files --others --exclude-standard
```

Expected: both commands list exactly the three finalized plan files.

Commit:

```bash
git add -- d4rt_agent/plans/phase_0.md
git add -- d4rt_agent/plans/phase_1.md
git add -- d4rt_agent/plans/phase_2.md
git diff --cached --name-status
git commit -m "docs(d4rt-agent): add orchestration and grounding tool plans"
```

Verify:

```bash
git status --short --branch
git show --stat --oneline HEAD
```

Expected: a clean feature branch whose only commit beyond `main` adds the three finalized plans.

## Step 0.10 — final Phase 0 audit

Run:

```bash
git log --oneline --decorate --graph --all -12
git rev-parse main
git rev-parse test/grounding
git rev-parse archive/failed-forced-grounding-prompt
git rev-parse feat/qwen-grounding-tool-orchestration
git diff --quiet main test/grounding
git status --short --branch
```

Required final state:

- `main` and `test/grounding` both point to `7e410864...`.
- The archive branch is ahead of `7e410864...` by three local commits.
- The feature branch is based on `7e410864...` and is ahead only by the finalized-plan commit.
- The feature worktree is clean.
- `git diff --quiet main test/grounding` exits 0.
- No push occurred.

Check that the failed result is still recoverable:

```bash
git show archive/failed-forced-grounding-prompt:d4rt_agent/results/dsi_bench/FINDINGS.md
```

Check that `main` still contains the previous 10/25 DSI result rather than the failed 7/25 result:

```bash
git show main:d4rt_agent/results/dsi_bench/DSI_BENCH_RESULTS.md
```

The main report should show 25 complete agent rows. The archive findings should show the failed forced-grounding experiment at 7/25 with four unanswered rows.

## Phase 0 completion criteria

Phase 0 is complete only when every item is true:

- [ ] The initial dirty state is preserved on `archive/failed-forced-grounding-prompt`.
- [ ] The archive branch contains separate implementation, evaluation, and plan commits.
- [ ] `main` contains exactly the four approved grounding-probe commits beyond `3172b84f...`.
- [ ] No archive-only commit is reachable from `main`.
- [ ] `test/grounding` is unchanged at `7e410864...`.
- [ ] `feat/qwen-grounding-tool-orchestration` starts from updated `main`.
- [ ] The feature branch contains the three finalized plan files and not the draft files.
- [ ] The current branch is `feat/qwen-grounding-tool-orchestration`.
- [ ] The worktree is clean.
- [ ] Nothing was pushed.

Do not start Phase 1 until this checklist passes.

## Recovery guidance

All recovery should be non-destructive:

- If a path is staged incorrectly before a commit, use `git restore --staged -- <exact-path>`.
- If a command fails before a commit, stop and inspect `git status`; do not reset.
- If the feature or archive branch name unexpectedly exists, inspect it with `git log` and ask for direction rather than deleting it.
- If a wrong commit was created, preserve it and add a corrective commit or ask before rewriting history.
- The archive branch is the authoritative recovery point for the failed experiment.
