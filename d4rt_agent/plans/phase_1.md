# Phase 1 — enforce one real action per model turn and make host evidence authoritative

## 1. Goal

Implement a generic tool-using conversation loop with this invariant:

```text
reason about current evidence
→ submit exactly one action
→ stop
→ host validates and executes that action
→ host returns the real result and authoritative evidence ledger
→ begin the next assistant turn
```

The model may continue thinking aloud before its action. The model must not be allowed to place imagined future actions or fabricated host results into the conversation it sees on later turns.

This phase fixes orchestration only. It does not add the separate Qwen grounding tool; that is Phase 2.

## 2. Why this phase is required

The current host already executes only the first JSON action in a Qwen response. The defect is the order of operations in `SimpleV2Orchestrator.solve`:

```text
generate the complete response
append the complete response to messages
extract the first action
execute only that action
append the real result
```

Qwen sometimes emits a complete imagined workflow in one generation:

```text
I should measure the runner.

{"action":"query_d4rt", ...}

Tool result d4rt_1: ...

I will calculate the displacement.

{"action":"python_math", ...}
```

Only `query_d4rt` is executed, but the fabricated result and future calculation remain in history. On the next turn the model treats its invented `d4rt_1`, `d4rt_2`, or `math_1` as real. The forced-grounding run then produced unknown-evidence loops, repeated rejected calls, four unanswered questions, and much longer runtime.

The fix must preserve:

- the model’s reasoning before its submitted action;
- the exact action that the host received;
- the complete raw response for diagnostics.

The fix must exclude from effective history:

- text after the first complete action;
- imagined tool results;
- second or later action objects;
- reasoning that assumes a not-yet-executed action succeeded.

## 3. Scope

### In scope

- A shared, generic one-action turn protocol.
- System-prompt examples formatted as explicit assistant and host turns.
- First-action extraction with exact character boundaries.
- History truncation after the first complete action.
- Complete raw-response retention in traces.
- Host-generated evidence ledgers after accepted and rejected turns.
- Error messages that name the requested unknown ID and list the IDs that actually exist.
- CPU unit and scripted-conversation tests.
- A small Blackwell replay of the three known pathological DSI questions.
- Scratch-only Phase 1 evaluation artifacts.

### Explicitly out of scope

Do not implement any of these in Phase 1:

- the separate Qwen bbox/points grounding action;
- token-level Transformers stopping criteria;
- automatic rejection of duplicate D4RT queries;
- DSI-specific partial-visibility policy changes;
- task-specific answer heuristics;
- a new forced-final-answer policy;
- removal or redesign of the strict/relaxed retry;
- changes to the 14-step budget;
- textual validation of quantitative prose;
- new camera-motion or facing-axis heuristics;
- any A100 evaluation.

The user wants the orchestration to remain generic. Do not solve individual DSI cases with benchmark-specific rules.

## 4. Prerequisites

Phase 0 must be complete.

Run:

```bash
git branch --show-current
git status --short --branch
git rev-parse main
git merge-base --is-ancestor main HEAD
```

Required state:

- branch: `feat/qwen-grounding-tool-orchestration`;
- clean worktree;
- local `main` includes commit `7e4108642f769a5db671e72a79338eabb30e83d7`;
- the feature branch descends from `main`.

Stop if any requirement fails. Do not implement Phase 1 on the archive branch, `test/grounding`, or `main`.

Run the existing CPU tests before editing:

```bash
/cluster/work/igp_psr/spanwar/envs/d4rt_bw/bin/python -m unittest \
  d4rt_agent.test_simple_v2 \
  d4rt_agent.test_dsi_bench \
  d4rt_agent.test_qwen_grounding_debug
```

Record the pass/fail count. Existing failures must be understood before new changes are attributed to this phase.

## 5. Existing code map

Read these files before editing:

| File | Current responsibility | Phase 1 change |
| --- | --- | --- |
| `d4rt_agent/simple_v2.py` | Qwen loading, action extraction, orchestration, evidence registry | Main implementation seam |
| `d4rt_agent/simple_v2_contracts.py` | Three action schemas and validation | Keep three tools; improve validation error context only if needed |
| `d4rt_agent/dsi_bench_run.py` | DSI subclass, strict/relaxed attempts, answer rules | Preserve retry behavior; inherit the corrected loop |
| `d4rt_agent/prompts/simple_v2_system.md` | Generic metric-task tool instructions/examples | Reformat examples into turns |
| `d4rt_agent/prompts/dsi_bench_system.md` | DSI reasoning instructions/examples | Remove failed forced-grounding changes and reformat examples |
| `d4rt_agent/test_simple_v2.py` | Contracts, fake Qwen/backend, orchestration tests | Add most new tests here |
| `d4rt_agent/test_dsi_bench.py` | DSI-specific host and prompt tests | Add prompt/integration assertions |
| `d4rt_agent/orchestration_audit.py` | New CPU-only trace invariant checker | Audit scratch and final answer directories |
| `d4rt_agent/test_orchestration_audit.py` | New synthetic audit tests | Cover valid and structurally invalid traces |
| `d4rt_agent/dsi_bench_report.py` | Trace rendering | Render effective response and discarded suffix diagnostics |
| `scripts/run_dsi_bench_blackwell.slurm` | Blackwell evaluation launcher | Add safe result/manifest/selection overrides |

The Phase 0 base deliberately excludes the failed prompt-only experiment. Do not copy its forced `GROUNDING`/`REUSE GROUNDING` declarations back into the prompt.

## 6. Required behavioral contract

### 6.1 One assistant turn

Every model turn has this form:

```text
Optional plain-text reasoning.

{"action":"ONE_ACTION","arguments":{...}}
```

The first complete action object is the submitted action. The effective assistant turn ends at that object’s closing brace.

### 6.2 Raw versus effective response

For every generation retain two concepts:

- `raw_qwen_response`: the exact complete decoded model output.
- `effective_response`: the prefix visible in later model history.

When an action is parsed:

```python
effective_response = raw_qwen_response[:extracted.end].strip()
discarded_suffix = raw_qwen_response[extracted.end:]
```

Do not reconstruct the action with `json.dumps`; doing so would lose the exact reasoning and formatting Qwen produced.

### 6.3 Host ownership

Only the host may:

- decide whether an action is valid;
- execute a tool;
- create an evidence ID;
- state a tool result;
- list available evidence.

The model may cite an ID only after the host has created it.

### 6.4 Evidence creation

- A successfully executed `query_d4rt` creates one `d4rt_N`.
- A successfully executed `python_math` creates one `math_N`.
- A valid tool result containing null/invisible measurements is still real evidence and receives an ID.
- A parse failure creates no evidence.
- A schema rejection creates no evidence.
- An evidence-reference rejection creates no evidence.
- A tool execution failure creates no evidence.
- A rejected action does not advance any evidence counter.
- `final_answer` terminates the task and does not become reusable evidence.

### 6.5 No hidden future state

Any raw suffix after the first action:

- is stored in the trace;
- is never appended to `messages`;
- is never parsed as another live action;
- is never treated as a host result;
- is never used to create evidence.

## 7. Add the shared tool-turn protocol

Create:

```text
d4rt_agent/prompts/tool_turn_protocol.md
```

Its content must be generic and include all of these statements:

1. Each assistant response is one turn.
2. The assistant may think aloud in plain text.
3. The assistant must choose exactly one action.
4. The response must end at the closing brace of that one action.
5. The assistant must wait for the host before choosing the next action.
6. The assistant must never simulate a tool result or write `Tool result`.
7. The assistant must never write a `HOST TURN`.
8. The assistant must never invent an evidence identifier.
9. The assistant must not put JSON objects in its prose reasoning.
10. The first action JSON is the action submitted to the host.
11. Only the host’s evidence ledger is authoritative.

Use this wording:

```text
## Tool interaction protocol

Each response is exactly one assistant turn.

During a turn:
1. You may think aloud in plain text.
2. Choose exactly one action.
3. End the response with exactly one JSON action.
4. Stop immediately after that action's closing brace.
5. Wait for the host response before deciding what to do next.

Never emit a second action, simulate a tool result, write a HOST TURN,
invent an evidence ID, continue reasoning after the action, or put JSON
objects in your prose reasoning.

The first action JSON is submitted to the host. Only the host creates tool
results and evidence identifiers. Only IDs in HOST EVIDENCE STATE exist.
```

Load this fragment in `simple_v2.py` next to the existing prompt loader. Compose it exactly once in `SimpleV2Orchestrator.__init__`:

```text
shared protocol

task-specific system prompt
```

Do not prepend it to the tool-free DSI baseline. The baseline calls `OfflineQwen.generate` directly and does not use `SimpleV2Orchestrator`.

Add a test proving that:

- the protocol is present exactly once in an orchestrator’s system message;
- it is not injected into `BASELINE_PROMPT`;
- a caller-provided `system_prompt` still appears after the shared protocol.

## 8. Reformat worked examples as explicit turns

The user chose complete multi-tool examples because isolated syntax examples do not teach the model how to combine query, calculation, and final answer.

Use this format:

```text
ASSISTANT TURN 1

Reasoning based only on evidence available before turn 1.

{"action":"query_d4rt","arguments":{...}}

--- END ASSISTANT TURN; STOP AND WAIT FOR HOST ---

HOST TURN

Tool result d4rt_1:
{...short illustrative result...}

HOST EVIDENCE STATE — authoritative

Available evidence:
- d4rt_1: query_d4rt

Last action: accepted
New evidence created: d4rt_1

--- END HOST TURN ---

ASSISTANT TURN 2

Reasoning based on the real result above.

{"action":"python_math","arguments":{...}}
```

Prompt-example requirements:

- Every assistant example contains exactly one action.
- Every assistant example ends immediately after the action.
- Only example host sections contain tool results and ledgers.
- Every cited ID was introduced in an earlier host section.
- At least one workflow reaches `final_answer`.
- Collectively the examples use all three Phase 1 tools:
  - `query_d4rt`;
  - `python_math`;
  - `final_answer`.
- Examples retain the useful DSI lessons about fixed/varying `t_cam`, range, object motion, and facing axes.
- Do not include the failed experiment’s forced visual grounding declaration.
- Keep example tool payloads short enough that the prompt does not grow substantially beyond the current prompt.

Update prompt tests so they split or parse each `ASSISTANT TURN` block and prove there is exactly one valid action in it. Do not merely search for keywords.

## 9. Replace action extraction with a boundary-aware result

### 9.1 Data type

In `d4rt_agent/simple_v2.py`, add:

```python
from dataclasses import dataclass
```

Then define:

```python
@dataclass(frozen=True)
class ExtractedAction:
    value: dict[str, Any]
    start: int
    end: int
```

Replace `_extract_action_json(text) -> dict` with:

```python
def extract_first_action(text: str) -> ExtractedAction:
    ...
```

If compatibility with private imports is needed, retain `_extract_action_json` only as a thin wrapper returning `.value`. Prefer updating internal tests/callers to the new public helper.

### 9.2 Parsing algorithm

Use `json.JSONDecoder.raw_decode`, not regular-expression brace matching:

```python
decoder = json.JSONDecoder()
for match in re.finditer(r"\{", text):
    try:
        value, consumed = decoder.raw_decode(text[match.start():])
    except json.JSONDecodeError:
        continue
    if isinstance(value, dict) and ("action" in value or "name" in value):
        return ExtractedAction(
            value=value,
            start=match.start(),
            end=match.start() + consumed,
        )
raise ValueError(...)
```

The returned `end` is an exclusive character offset into the original raw string.

### 9.3 Required extraction tests

Add tests for:

- reasoning followed by one action;
- leading/trailing whitespace;
- an action inside a Markdown code fence;
- nested dictionaries and arrays;
- braces inside a JSON string;
- escaped quotes inside a JSON string;
- a non-action JSON object before the action;
- two valid actions: the first action and its first boundary must win;
- fake `Tool result` text after the action;
- malformed JSON before a valid action;
- no action at all;
- exact reconstruction: `raw[:end]` equals reasoning plus the first complete action and excludes the next character.

Do not implement a simplistic “find the last brace” solution. It will fail on nested structures and strings.

## 10. Correct the orchestration order

Refactor `SimpleV2Orchestrator.solve` into explicit stages. Do not append the raw model response to `messages` before extraction.

### 10.1 Required order

For each step:

1. Generate `raw`.
2. Start a trace attempt with `raw_qwen_response`.
3. Extract the first action and boundary.
4. Derive `effective_response` and `discarded_suffix`.
5. Append only `effective_response` to assistant history.
6. Validate the parsed action.
7. Validate prior-evidence references as part of normal execution.
8. Execute exactly one action.
9. Create evidence only after successful execution.
10. Create a host ledger snapshot.
11. Append the real result, ledger, and optional step warning as the next user/host message.
12. Generate the next assistant turn.

### 10.2 Pseudocode

Use this as the implementation skeleton:

```python
for step in range(1, self.max_steps + 1):
    raw = self.qwen.generate(messages)
    attempt = {
        "step": step,
        "kind": "model_action",
        "raw_qwen_response": raw,
    }

    try:
        extracted = extract_first_action(raw)
    except ValueError as error:
        attempt.update(
            status="rejected",
            failure_stage="parse",
            error=str(error),
            effective_response=raw,
            discarded_suffix="",
            parsed_action=None,
        )
        trace.append(attempt)
        messages.append(assistant_message(raw))
        ledger = ledger_after_rejection(...)
        messages.append(host_rejection_and_ledger(error, ledger, notice))
        continue

    effective = raw[:extracted.end].strip()
    discarded = raw[extracted.end:]
    attempt.update(
        effective_response=effective,
        discarded_suffix=discarded,
        action_span={"start": extracted.start, "end": extracted.end},
    )
    messages.append(assistant_message(effective))

    try:
        name, arguments = validate_action(extracted.value)
        attempt["parsed_action"] = {
            "action": name,
            "arguments": arguments,
        }
        ...
    except (KeyError, TypeError, ValueError) as error:
        attempt.update(
            status="rejected",
            failure_stage=current_stage,
            error=str(error),
        )
        trace.append(attempt)
        ledger = ledger_after_rejection(...)
        messages.append(host_rejection_and_ledger(error, ledger, notice))
        continue
```

`current_stage` must distinguish at least:

- `parse`;
- `validation`;
- `execution`;
- `final_validation`.

All may use trace status `rejected`; `failure_stage` supplies the diagnostic detail. A tool exception that is not an expected validation/rejection must still propagate to the outer benchmark error handler rather than being silently converted into a model mistake.

### 10.3 No-action behavior

If there is no parseable action:

- save the complete raw response;
- keep the model’s plain-text response as `effective_response`;
- append it as the assistant turn;
- create no evidence;
- append a host rejection and ledger;
- instruct the model that its next response must end in one action.

Do not attempt to execute prose or infer the intended action.

### 10.4 Final-answer behavior

For a valid `final_answer`:

- run `_validate_final`;
- save the raw/effective/discarded fields;
- record status `ok`;
- record a final ledger snapshot in the trace;
- return immediately;
- do not append a fabricated host result after termination.

If final validation fails:

- no final counter/evidence is created;
- append the rejection and current ledger;
- allow the next turn.

## 11. Add an authoritative evidence ledger

### 11.1 Internal representation

Keep the existing `evidence: dict[str, dict[str, Any]]` result registry for backward compatibility.

Add metadata maintained by the host:

```python
@dataclass(frozen=True)
class EvidenceEntry:
    evidence_id: str
    tool_name: str
    created_at_step: int
```

Store entries in insertion order. Never infer tool type by trusting model text.

### 11.2 Model-facing format

After a successful evidence-producing action:

```text
HOST EVIDENCE STATE — authoritative

Available evidence:
- d4rt_1: query_d4rt
- math_1: python_math

Last action: accepted
New evidence created: math_1

Only identifiers listed under Available evidence exist.
```

After rejection:

```text
HOST EVIDENCE STATE — authoritative

Available evidence:
- d4rt_1: query_d4rt

Last action: rejected
New evidence created: none
Reason: Evidence ID 'd4rt_2' does not exist.

Rejected actions create no evidence identifiers.
Only identifiers listed under Available evidence exist.
```

For an empty registry:

```text
Available evidence:
- none
```

The ledger must:

- be generated entirely from host state;
- appear after the real result or rejection;
- remain concise;
- contain IDs and tool names, not duplicate full payloads;
- never contain benchmark-specific recovery advice;
- state whether a new ID was created;
- be saved as structured metadata in the trace as well as formatted text in the conversation.

### 11.3 Trace ledger snapshot

Each trace row should contain:

```json
{
  "evidence_state": {
    "available": [
      {
        "evidence_id": "d4rt_1",
        "tool_name": "query_d4rt",
        "created_at_step": 1
      }
    ],
    "last_action": "accepted",
    "new_evidence_id": "d4rt_1",
    "reason": null
  }
}
```

For rejection, `new_evidence_id` is null and `reason` contains the exact error returned to the model.

## 12. Make errors actionable and truthful

Create one generic helper for unknown evidence, for example:

```python
def unknown_evidence_error(
    requested: Sequence[str],
    evidence: Mapping[str, Any],
) -> str:
    requested_text = sorted(set(requested))
    available = sorted(evidence)
    return (
        f"Evidence ID(s) {requested_text!r} do not exist. "
        f"Available evidence IDs: {available}. "
        "Rejected actions create no evidence. Use an available ID, or submit "
        "the missing tool call as this turn's single action."
    )
```

Use the helper in:

- `resolve_bindings`;
- `_validate_final_evidence`;
- `DSIOrchestrator._validate_final`;
- any other evidence lookup discovered with:

```bash
rg -n "unknown evidence|not in evidence|evidence_id" d4rt_agent
```

Requirements:

- Name the invalid ID.
- List all actual available IDs, including an empty list when none exist.
- State that rejected actions create no ID.
- Give one generic correction: use an existing ID or submit the missing call.
- Never claim an ID exists because it appeared in Qwen’s raw response.

Retain the existing helpful DSI math/visibility diagnosis. This phase does not remove it.

## 13. Trace schema and reporting

Every model turn must contain:

```json
{
  "step": 1,
  "kind": "model_action",
  "raw_qwen_response": "complete decoded output",
  "effective_response": "reasoning plus first action",
  "discarded_suffix": "everything after the first action",
  "action_span": {"start": 35, "end": 211},
  "parsed_action": {
    "action": "query_d4rt",
    "arguments": {}
  },
  "status": "ok",
  "failure_stage": null,
  "call_id": "d4rt_1",
  "result": {},
  "evidence_state": {}
}
```

For parse failure, `parsed_action` and `action_span` may be null. Do not omit the raw/effective/discarded keys.

Update `d4rt_agent/dsi_bench_report.py` so:

- the normal trace view reflects `effective_response`, because that is the real conversation;
- the action table continues to use `parsed_action`;
- diagnostic details can show `raw_qwen_response` and `discarded_suffix`;
- old result records without new fields remain readable;
- discarded suffixes are HTML-escaped or fenced safely in Markdown;
- reports indicate how many turns had non-whitespace discarded suffixes and how many suffixes contained another parseable action.

Do not regenerate or overwrite canonical DSI results in Phase 1.

## 14. Tests

### 14.1 Extend the scripted Qwen

The existing `_ScriptedQwen` in `test_simple_v2.py` stores messages. Extend or reuse it so tests can inspect the exact messages supplied to each generation.

### 14.2 Boundary tests

Implement every extraction test in Section 9.3.

### 14.3 Successful multi-turn conversation

Script these Qwen responses:

1. reasoning + valid `query_d4rt` + fake tool result + future `python_math`;
2. reasoning + valid `python_math` + fake future `final_answer`;
3. valid `final_answer`.

Assert:

- the backend query executes once after response 1;
- response 2 receives only response 1’s effective prefix, real tool result, and ledger;
- response 2 does not contain the fabricated result or future calculator from response 1 anywhere in its input messages;
- response 3 receives the real math result;
- the final result terminates;
- raw suffixes are preserved in trace;
- exactly `d4rt_1` and `math_1` exist;
- no phantom `d4rt_2` or `math_2` exists.

### 14.4 Rejection and counter tests

Test:

- invalid schema then valid D4RT query produces `d4rt_1`, not `d4rt_2`;
- unknown math evidence then corrected binding produces `math_1`, not `math_2`;
- failed final validation does not consume `final_1`;
- a tool execution `ValueError` creates no evidence;
- a valid D4RT response with all visibility false still creates evidence;
- existing evidence remains unchanged after rejection;
- ledger IDs exactly match `evidence.keys()` after every row;
- ledger tool types match the action that created each ID;
- rejection messages list the actual available IDs.

### 14.5 No-action tests

Test a plain prose response with no action:

- it is stored as raw and effective text;
- no evidence is created;
- a parse-stage rejection is appended;
- the next turn receives the authoritative empty ledger;
- a later corrected action can succeed.

### 14.6 Prompt tests

For both tool-using prompts, assert:

- the shared protocol says one action and wait;
- simulated tool results are forbidden;
- assistant examples are explicitly separated from host examples;
- no assistant example contains two action objects;
- only host examples introduce evidence IDs;
- at least one example composes query → math → final;
- the failed forced-grounding `GROUNDING` declaration is absent.

### 14.7 Backward compatibility tests

- Existing three action schemas remain the only schemas in Phase 1.
- `point_mode` remains absent from all model-facing schemas and messages.
- Existing trace replay still accepts older completed artifacts.
- Existing report generation can read old records without `effective_response`.
- Tool-free baseline behavior is unchanged.

## 15. CPU validation sequence

Run focused tests first:

```bash
/cluster/work/igp_psr/spanwar/envs/d4rt_bw/bin/python -m unittest \
  d4rt_agent.test_simple_v2.OrchestratorTest
```

Then prompt and DSI host tests:

```bash
/cluster/work/igp_psr/spanwar/envs/d4rt_bw/bin/python -m unittest \
  d4rt_agent.test_dsi_bench
```

Then the complete relevant suite:

```bash
/cluster/work/igp_psr/spanwar/envs/d4rt_bw/bin/python -m unittest \
  d4rt_agent.test_simple_v2 \
  d4rt_agent.test_dsi_bench \
  d4rt_agent.test_qwen_grounding_debug \
  d4rt_agent.test_orchestration_audit
```

Also compile the edited modules:

```bash
/cluster/work/igp_psr/spanwar/envs/d4rt_bw/bin/python -m compileall -q d4rt_agent
```

Do not submit a GPU job while any CPU test fails.

## 16. Make the Blackwell launcher safely reusable

Edit only `scripts/run_dsi_bench_blackwell.slurm` for evaluation. Do not invoke the A100 launcher.

Add environment overrides with current behavior as defaults:

```text
RESULTS_DIR
MANIFEST_PATH
ONLY_FILE
FORCE
```

Required semantics:

- `RESULTS_DIR` defaults to `d4rt_agent/results/dsi_bench`.
- `MANIFEST_PATH` defaults to the canonical manifest under the repository.
- `ONLY_FILE`, when set, must exist and contain one question ID per non-empty line.
- `FORCE=1` adds `--force`; absent or `0` does not.
- The launcher always passes `--manifest "$MANIFEST_PATH"`.
- The launcher prints all four effective values before model loading.
- Empty lines in `ONLY_FILE` are ignored.
- Duplicate IDs in `ONLY_FILE` are rejected or de-duplicated deterministically.
- Any unknown `FORCE` value is rejected.
- Existing `MODE`, `LIMIT`, and `POINT_MODE` behavior remains valid.
- The hardcoded SLURM request remains:
  - partition `cuda13pr.4h`;
  - `nvidia_rtx_pro_6000:1`;
  - constraint `EPYC_9654`.

Add a lightweight shell syntax check:

```bash
bash -n scripts/run_dsi_bench_blackwell.slurm
```

Do not change the launcher to A100 and do not remove the Blackwell constraint.

## 17. Add a generic audit utility

Add a small CPU-only audit command rather than relying on manual JSON inspection. Use:

```text
d4rt_agent/orchestration_audit.py
```

It should accept:

```text
--answers-dir PATH
--output PATH
```

It should report:

- number of records and statuses;
- strict versus relaxed attempts;
- total model turns;
- successful and rejected actions by type;
- rejection counts by `failure_stage`;
- turns with discarded suffixes;
- discarded suffixes containing another action;
- evidence IDs created;
- unknown-evidence rejection count;
- maximum steps;
- any ledger/registry mismatch;
- any accepted action that references unavailable evidence;
- any record where effective history contains more than one action.

The audit must never load Qwen or D4RT. It should return non-zero for structural invariant violations and zero when the trace contract is satisfied. Accuracy is not a structural violation.

Add CPU tests using small temporary answer records.

The command and its tests are required before the pathology replay. Do not replace the audit with manual inspection.

## 18. Blackwell pathology replay

### 18.1 Fixed questions

Use exactly these three question IDs:

```text
CameraBench_c2_camerabench_u35wis62r2m.1.4_8c093924
k700_c4_k700_jogging_qy8rjbxblna_000116_000126_video-scene-00001_ab85ad6e
llava178k_c5_videos_youtube_video_2024_ytb_16mzjk4aojs_video-scene-00000_b36d2eb1
```

They correspond to prior cases #3, #17, and #22:

- phantom `d4rt_2`;
- repeated rejected calculations and nonexistent evidence;
- a valid first action followed by a generated invalid future action.

### 18.2 Scratch location

All Phase 1 GPU outputs must remain under:

```text
/cluster/scratch/spanwar/tmp/d4rt_agent/phase_1_orchestration
```

Use a fresh run subdirectory. Never point the Phase 1 job at `d4rt_agent/results/dsi_bench`.

Create:

```text
/cluster/scratch/spanwar/tmp/d4rt_agent/phase_1_orchestration/question_ids.txt
```

with exactly the three IDs above, one per line. Verify:

```bash
sed -n '1,10p' /cluster/scratch/spanwar/tmp/d4rt_agent/phase_1_orchestration/question_ids.txt
wc -l /cluster/scratch/spanwar/tmp/d4rt_agent/phase_1_orchestration/question_ids.txt
sort -u /cluster/scratch/spanwar/tmp/d4rt_agent/phase_1_orchestration/question_ids.txt
```

Use `apply_patch` or a normal editor to create this file; do not generate IDs from the current answer directory. Expected:

- the first command prints the three IDs in Section 18.1 and nothing else;
- the line count is `3`;
- the sorted unique output also contains three lines.

In one shell, create a collision-free run directory:

```bash
PHASE1_RUN_DIR="/cluster/scratch/spanwar/tmp/d4rt_agent/phase_1_orchestration/run_$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=12 HEAD)"
test ! -e "$PHASE1_RUN_DIR"
mkdir -p "$PHASE1_RUN_DIR"
cp -- d4rt_agent/results/dsi_bench/manifest.json "$PHASE1_RUN_DIR/manifest.json"
```

If `test ! -e` fails, choose a new run name. Do not delete or reuse the existing directory.

Copy the canonical manifest into the fresh run directory and verify that the copies are byte-identical:

```bash
sha256sum d4rt_agent/results/dsi_bench/manifest.json
sha256sum "$PHASE1_RUN_DIR/manifest.json"
cmp -- d4rt_agent/results/dsi_bench/manifest.json "$PHASE1_RUN_DIR/manifest.json"
```

Expected: identical hashes and `cmp` exit code 0.

### 18.3 Submit only to Blackwell

Submit:

```bash
sbatch --parsable \
  --export=ALL,POINT_MODE=ensemble5,RESULTS_DIR="$PHASE1_RUN_DIR",MANIFEST_PATH="$PHASE1_RUN_DIR/manifest.json",ONLY_FILE=/cluster/scratch/spanwar/tmp/d4rt_agent/phase_1_orchestration/question_ids.txt,FORCE=0 \
  scripts/run_dsi_bench_blackwell.slurm
```

Keep this in the same shell where `PHASE1_RUN_DIR` was set. Record:

- job ID;
- current Git commit;
- scratch results path;
- submission command.

Inspect allocation after scheduling:

```bash
scontrol show job <job-id>
```

The job must request `nvidia_rtx_pro_6000` and constraint `EPYC_9654`. Cancel and resubmit correctly if it is routed to an A100.

Do not busy-wait. Use SLURM status/log inspection at meaningful intervals:

```bash
squeue -j <job-id>
sstat -j <job-id>.batch
tail -n 80 logs/d4rt_agent/dsi_bench_<job-id>.out
```

### 18.4 Required pathology-run gate

All of these must pass:

- exactly three answer JSONs exist;
- all three IDs match the fixed list;
- all three strict attempts terminate within 14 steps;
- no row contains `gated_attempt`;
- no row is `complete_relaxed`;
- no effective response contains more than one action;
- fabricated `Tool result` text may exist only in `discarded_suffix`, never later model-visible history;
- every ledger exactly matches the evidence registry at that step;
- no accepted action cites a nonexistent ID;
- no repeated unknown-ID rejection loop occurs;
- all raw generations remain available for diagnostics.

Record accuracy, but do not use accuracy as a Phase 1 pass/fail criterion. The experiment isolates orchestration correctness.

If any structural gate fails:

1. Do not proceed to Phase 2.
2. Save the scratch records and log.
3. Add a minimal reproducing CPU test.
4. Fix the generic orchestration defect.
5. Rerun CPU tests.
6. Submit a new fresh pathology run.

Do not patch the prompt with an example tailored to one failed question.

## 19. Commit strategy

Keep commits reviewable. Use this sequence:

1. `feat(d4rt-agent): enforce one action per model turn`
   - boundary-aware extractor;
   - corrected history insertion;
   - raw/effective/discarded trace fields;
   - unit/scripted tests.
2. `feat(d4rt-agent): add authoritative evidence ledger`
   - ledger metadata and formatting;
   - actionable unknown-evidence errors;
   - rejection/counter tests.
3. `docs(d4rt-agent): teach explicit assistant-host tool turns`
   - shared protocol;
   - reformatted simple and DSI examples;
   - prompt tests.
4. `chore(d4rt-agent): add scratchable Blackwell orchestration replay`
   - launcher overrides;
   - audit utility and tests.

Before each commit:

```bash
git diff --check
git status --short
git diff --cached --name-status
```

Do not commit Phase 1 scratch results. Do not push.

## 20. Phase 1 completion criteria

Phase 1 is complete only when:

- [ ] Phase 0’s branch and history checks still hold.
- [ ] The shared protocol is host-owned and prepended once.
- [ ] Prompt examples are explicit assistant/host turns and teach all three tools.
- [ ] `extract_first_action` returns value, start, and end.
- [ ] Only reasoning plus the first action enters effective history.
- [ ] The full raw response and discarded suffix remain in traces.
- [ ] Every success/rejection has a structured authoritative ledger snapshot.
- [ ] Rejections and execution failures create no evidence or counter gaps.
- [ ] Unknown-evidence errors list the IDs that actually exist.
- [ ] Existing CPU tests and all new tests pass.
- [ ] The Blackwell launcher passes `bash -n`.
- [ ] The three pathology questions were run on an RTX PRO 6000 Blackwell node.
- [ ] All pathology structural gates pass.
- [ ] Phase 1 outputs remain retrievable in the selected scratch directory.
- [ ] Canonical 25-question results were not overwritten.
- [ ] No A100 job was used.
- [ ] No result was pushed.

Only after this checklist passes should Phase 2 begin.

## 21. Final design principle

The model owns reasoning and tool choice. The host owns turn boundaries, execution, and evidence truth.

```text
Think → submit one action → stop → receive reality → think again
```
