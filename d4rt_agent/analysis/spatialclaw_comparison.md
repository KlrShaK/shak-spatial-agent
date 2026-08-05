# SpatialClaw vs. D4RT Agent — ACI and Prompt-Architecture Review

**Date:** 2026-08-05
**Reviewed:** `NVlabs/SpatialClaw` @ `b062f82` (initial public release + 2 commits) vs.
`d4rt_agent/` @ `archive/failed-forced-grounding-prompt` (`2068422`)
**Frame:** post-training / agent-engineering review. Every recommendation is tied to a
specific prompt-engineering principle *and* to evidence already in this repo's own runs.

---

## 0. TL;DR

SpatialClaw and the D4RT agent are solving the same problem — a VLM that cannot measure 3D,
plus tools that can — and they made **opposite bets on where structure is enforced.**

| | D4RT agent | SpatialClaw |
|---|---|---|
| Interface | JSON action in free text, schema pasted into a user message | Free-form Python against a persistent IPython kernel |
| Structure enforcement | prompt text + post-hoc validation | prompt text + post-hoc regex + **typed values + poka-yoked tools** |
| Constrained decoding | none | none |
| Tool results | 3.4 KB raw JSON envelope per call | `__repr__` that *teaches its own next call* |
| Grounding | model writes a box in the action, from full context | isolated `vlm.locate` session with a refusal preamble |
| Prompt | one 28 KB monolith | ~13 named, individually ablatable sections |
| History | full, never truncated | full, never truncated, **except** retroactive error condensation |

The central finding: **your failures are not reasoning failures, they are authority
failures.** The model writes text that the host then treats as reality. SpatialClaw's
design makes that structurally hard in three places where yours makes it easy. That is
where the transferable value is — not in the prompt wording.

Second finding: the forced-grounding experiment failed for a reason the literature
predicts, and the fix SpatialClaw already implements is *the same intervention moved out of
the prompt and into the tool layer*. Your own grounding-debug data says it should work.

---

## 1. The bug that dominates everything else

### 1.1 What the code does

`simple_v2.py:381-382` appends the **entire raw model response** to history:

```python
raw = self.qwen.generate(messages)
messages.append({"role": "assistant", "content": [{"type": "text", "text": raw}]})
```

`_extract_action_json` (`simple_v2.py:90-101`) then executes only the **first** JSON object:

```python
for match in re.finditer(r"\{", text):
    ...
    if isinstance(value, dict) and ("action" in value or "name" in value):
        return value
```

Meanwhile `dsi_bench_system.md:284` promises the opposite contract:

> The response must end with exactly one JSON action and nothing after it

So when the model emits *action → invented tool result → second action*, the host runs
action 1, discards the rest — **and leaves the invention in the transcript as the model's
own words.** On the next turn the model reads its own fabrication as fact.

### 1.2 Measured, in your traces

Re-derived directly from `results/dsi_bench/answers/*.json` (216 responses carrying raw text):

| Signal | Count | Rate |
|---|---:|---:|
| Responses containing >1 `"action"` | 59 | **27.3%** |
| Responses literally impersonating the host (`Tool result d4rt_N: {...}`) | 15 | **6.9%** |
| Rejections citing the nonexistent `d4rt_2` | **84** | dominant class |
| Rejections: "Qwen did not emit one JSON action: 'Tool result d4rt_2…'" | 13 | |

84 + 13 + 6 = **103 of ~120 rejections in this run are one bug.** Not a prompt-quality
issue; a transcript-authority issue.

### 1.3 Why this is the principle violation

Anthropic's *Building Effective Agents* defines the agent loop by exactly the property this
breaks: the agent **"obtains ground truth from the environment at each step."** Here the
environment's output and the model's output occupy the same undifferentiated text channel,
and the model has learned to write in the environment's voice. That is not a model defect —
you *taught* it the envelope. Every real result ships a 1.4 KB `sampled_to_original` table,
identical on all 14 calls; the fabricated results reproduce it verbatim.

### 1.4 SpatialClaw's structural answer

Three independent mechanisms, none of which is a prompt instruction:

1. **The model's channel and the host's channel are different data types.** The model emits
   markdown; the kernel returns a `FeedbackCollector`-built block. There is no string the
   model can write that becomes a variable.
2. **Out-of-band signalling with removal.** `show()` prints `[SHOW:{json}]` to stdout,
   `execute_node` parses it (`_SHOW_RE`), then `_strip_show_markers` deletes it so the model
   never sees the protocol it just triggered (`execute_node.py:19-48`). The model cannot
   imitate a marker it has never observed.
3. **Malformed output is never echoed** (`llm_step_node.py:78-85`):
   ```python
   # Don't echo the raw malformed output — it's noise and bloats context.
   ai_msg = AIMessage(content=f"[Format Error] {error_msg}")
   ```

### 1.5 Fix — and you have a cheaper one than SpatialClaw

You run HF `transformers` with `do_sample=False` (`simple_v2.py:269-292`), which means you
control decoding. Two changes, in order of cost:

**(a) Truncate history to the executed action.** One line. Append the reasoning prefix +
the extracted action, not `raw`. Removes the fabrication from context even when generated.

**(b) Stop generation at the first complete JSON object.** A `StoppingCriteria` that tracks
brace depth after the first `{` containing `"action"`. The model then *physically cannot*
emit a phantom result. This is strictly stronger than any prompt rule, costs one class, and
saves tokens on every call.

This is the one place you can be **better than SpatialClaw**, which relies entirely on
prompt compliance + regex and carries 4-tier and 3-tier parser fallback ladders as a result.
Constrained decoding deletes those ladders. Their `client.py:457-492` sends no
`response_format` and no `guided_json` — a genuine gap in an otherwise careful system.

---

## 2. Why forced grounding failed, and what to do instead

### 2.1 The diagnosis in your own findings is right, and generalizable

`results/dsi_bench/FINDINGS.md` records the outcome: 7/25 vs. 10/25 before, rejections
24 → 149, wall time 4.1× slower, three previously-correct answers lost to non-termination.
Compliance with the new protocol was **63/105 (60%)**, while citation of a grounding ID was
105/105 — i.e. the model learned the *surface token* and not the *behaviour*.

The prompt says the quiet part itself (`dsi_bench_system.md:280-282`):

> This is a reasoning protocol for you, not a host validation rule. Follow it even though
> the tool schema cannot enforce it.

That sentence is the experiment's own falsification. **An 8B model's instruction-following
is a budget, not a boolean.** You added ~1.5 KB of format obligation to a 28 KB prompt at a
768-token generation cap, and compliance was drawn from the same pool that was paying for
action formatting and termination. Both regressed. This is the standard failure mode of
prompt-stacking on a saturated instruction budget, and it is why the result was not
"grounding didn't help" but "everything else got worse."

### 2.2 The signal that says the idea was right

Two pieces of your own evidence:

- Matched-subset IoU vs. the standalone detector rose **0.013 → 0.263** on the 22
  exactly-matched `(question, frame, label)` keys.
- `results/qwen_grounding_debug/FINDINGS.md`: the *same model*, given one image and one
  detection prompt with **no** question, no other frames, no schema, no history, produces
  boxes that are visually on target where the in-loop boxes are displaced — and returns
  explicit `null` on 6 unlocatable landmarks rather than guessing.

Conclusion: **grounding degrades with context, not with capability.** The in-loop box is bad
because it is drawn under 32 images + 7k tokens of policy + a partially-fabricated
transcript, not because Qwen3-VL cannot localize.

### 2.3 SpatialClaw already built the fix

`vlm.locate` is a **separate tool with an isolated session** (`kernel_types/vlm_module.py:1-16`):

> Each method opens an isolated VLM session: the VLM sees only the images and the question —
> no conversation history, no prior reasoning from the main agent.

with refusal engineered in at two layers. Prompt-side (`vision_prompt.py:26`):

> Refusal is a valid, expected answer. Returning a "best-guess" coordinate near a
> similar-but-wrong object … counts as hallucination and is incorrect — refuse instead.

and code-side, prepended to **every** locate call regardless of what the agent asked
(`vlm_module.py:32-40`):

```python
_LOCATE_VERIFY_PREAMBLE = (
    "Before answering, first check whether what the question describes is "
    "clearly visible in the provided image(s). If it is absent or ambiguous, "
    "reply EXACTLY `Not visible` on its own line ... do NOT return coordinates "
    "in that case. Only if the requested target is clearly and unambiguously "
    "present, answer the request below.\n\nRequest: "
)
```

Plus an anti-priming rule the agent cannot bypass (`prompt_common.py:201`):

> The session is independent … Do not write your prior conclusions, prior vlm answers, or
> expected answers into the question — phrase the question on its own terms so the answer is
> formed from the images.

And the main prompt forbids the thing your agent does by default (`prompt_common.py:146`):

> Estimating coordinates yourself by eye is unreliable — call `vlm.locate` for grounding.

**This is your failed experiment, implemented correctly:** grounding is not a protocol the
model must remember to follow, it is a call it must make because there is no other way to
obtain a box. Context isolation is the mechanism; refusal-as-valid-output is what converts
your 54% invisible-target rate from a silent wrong answer into a recoverable one.

### 2.4 Recommendation

Add `locate_target(label, frame)` as a **fourth action**, implemented as a fresh
single-image Qwen call with your `qwen_grounding_debug.py` prompt (which already works),
returning `{"bbox_2d_1000": [...] | null}`. Then remove `bbox_2d_1000` from `query_d4rt`'s
schema and have it take a `grounding_id` instead. The box becomes unforgeable, cacheable by
`(label, t_src)`, and the "one declaration per unique pair" rule your prompt could not
enforce becomes a host invariant for free.

This matches `plans/phase_2.md`'s isolated grounding tool — and `plans/phase_1.md` is
correct that §1 must land first.

---

## 3. Tool results: 3.4 KB of JSON vs. a self-documenting repr

### 3.1 What your model receives

Measured over 95 real `d4rt_*` results: median redacted payload **3,407 chars**, max 6,689,
of which `sampled_to_original` is **1,422 chars — constant on every call.** Across 14 steps
that is ~20 KB of context spent restating a fixed mapping, inside a context that also holds
32 images.

### 3.2 What SpatialClaw's model receives

`__repr__` is written as a *mini-tutorial instantiated with the object's own indices*
(`per_frame_types.py:353-364`):

```python
return (
    f"PerFrameMask(frames={fi_str}, objects={self.labels}, shape={self.masks.shape})\n"
    f"  Use: seg.get_mask(frame={fi[0]}, object=0)  # (H, W) bool\n"
    f"  Use: seg.get_centroid_3d(recon, frame={fi[0]}, object=0)  # (3,) float32\n"
    f"  Use: seg.visualize({fi[0]})  # VisualFeedback"
)
```

A `seg` with `frame_indices=[15]` prints `seg.visualize(15)` — the next correct call, with
the right literal, in the observation. That is the *"formats close to naturally occurring
text"* and *"is it obvious from the description?"* guidance applied to **return values**,
which is where almost nobody applies it.

Images are handled the same way: `VisualFeedback.__repr__` yields a text surrogate
(source/description/size), `GraphDrawer.plot` yields a statistical surrogate
(`min/max/mean/first/last/trend`), and `render_bev` yields a semantic one — so a plot has a
cheap textual channel even when it is not shown.

### 3.3 Recommendation

Replace the raw `json.dumps(result)` feedback with a rendered summary:

```
d4rt_1: label="runner" src=f0 cam=f0  |  8 targets, 5 visible (63%)
  frames  [0, 4, 8, 12, 16, 20, 24, 28]
  visible [T, T, T, F, F, T, T, T]
  xyz     f0 (0.12, -0.03, 2.41) ±0.05   f4 (0.19, -0.02, 2.38) ±0.06   ...
  noise floor: median std/|z| = 0.03      <- differences below this are not measurable
  next: python_math with bindings {"traj": {"evidence_id":"d4rt_1","path":["math_trajectory_aligned_xyz_m"]}}
```

Drop `sampled_to_original` entirely (state it once, in the system prompt). Three wins:
the envelope is no longer imitable, the `bindings` shape — your **most-rejected action**,
95 invalid `python_math` calls, taught only in prose at `dsi_bench_system.md:304-316` — now
appears in the observation next to the ID it applies to, and you recover ~20 KB of context.

The noise-floor line is the second-order win. `dsi_bench_system.md:50-52` calls the std
*"**This is your noise floor**"*, and only **2 of 10** `python_math` calls in the pilot ever
referenced a `_std_` field. A rule stated once at 7k tokens' distance loses to a value
printed next to the number it qualifies.

---

## 4. Poka-yoke: the tool layer as error prevention

This is SpatialClaw's strongest area and maps directly onto Appendix 2 of *Building
Effective Agents* ("restructure arguments to prevent errors"; the SWE-bench absolute-path
example).

**Input coercion instead of rejection** — `_normalize_points_per_object`
(`sam3_tool.py:17-68`) fixes known LLM shapes and *tells the model it did*:

```python
"""Handles common LLM mistakes:
   - [x, y] (flat scalars) → [[[x, y]]]
   - [[x, y]] (missing object nesting) → [[[x, y]]]
   - [[[x], [y]]] (split coords) → [[[x, y]]]
"""
print(f"[SAM3] Auto-wrapped flat [x, y] into [[[x, y]]]: {result}")
```

**Every import style works** (`init_node.py:325-328`): `tools` is registered in
`sys.modules`, per-tool submodules too, *and* every tool is bound at top level, explicitly
so *"it survives even if the LLM accidentally reassigns 'tools'."*

**Types carry the argument most likely to be wrong.** `FrameImage.frame_index` means
`frame_indices=` never has to be passed.

**Errors are recovery menus, not diagnostics** (`sam3_tool.py:595-606`):

```
SAM3.segment_image_by_text(prompt=...) produced no usable mask on this image
(frame_index=..., confidence_threshold=...). SAM3 either detected nothing matching the
prompt, or its mask logits collapsed to all background. Try one of:
  - a different frame from the same video (this frame may be in a transient state),
  - a more specific prompt (e.g. an attribute + noun),
  - lowering confidence_threshold,
  - segment_image_by_box(image, [x1, y1, x2, y2]) with a known box location.
```

and the frame-cap error **computes the fix** (`reconstruct_tool.py:160-164`):

```python
n = len(frames); step = n // self.MAX_FRAMES + 1
raise ValueError(f"... Subsample first, e.g.:\n"
                 f"  recon = tools.Reconstruct.Reconstruct(InputImages[::{step}])")
```

**NaN, not None** — `centroid` returns `(nan, nan)` and `get_centroid_3d` returns
`[nan,nan,nan]`, documented as *"safe for arithmetic and formatting"*. A whole class of
`TypeError: unsupported operand type(s) for -: 'NoneType'` never happens.

**Credit where due:** your host messages are already in this style — `_diagnose_math`
rewriting "non-finite values" into a visibility explanation naming the frame
(`dsi_bench_run.py:206-250`), and `_SAFE_FUNCTION_HINT`. Several carry the SLURM job that
motivated them. That is better documented than most production agent code.

**The gap is the 54% invisible-target rate.** That is your `segment_image_by_text`-returns-
nothing case, and it currently comes back as `"visible": false` inside a 3.4 KB blob with no
menu. Give it one: *"target invisible at 5/8 frames. Try: a different `t_src` where the
target is larger; a nearer `t_cam`; or answer from the 3 visible frames and state the
limitation."* Your findings show the agent instead falls back to the raw images — becoming,
in your own words, *"a distractor rather than a channel."*

---

## 5. Prompt architecture: monolith vs. ablatable sections

`build_system_prompt()` (`system_prompt.py:332-351`) composes ~13 **named** sections through
`resolve_section`, which supports per-section exclusion and file override
(`prompt_common.py:13-39`), with typo warnings on unknown names. Tool docs are not in the
prompt module at all — each tool owns its `TOOL_PROMPT_DESCRIPTION`, sub-sectioned and
independently ablatable (`tool_sam3_api`, `tool_sam3_verify`, `tool_reconstruct_bev`, …).

Two consequences worth stealing:

1. **The prompt is a research object.** You can ablate `coordinate_system` or
   `evidence_hierarchy` and attribute the delta. Your forced-grounding trial changed the
   grounding block *and* rewrote all three worked examples in one commit — so when the
   result came back worse, the design could not tell you which half did it. Sectioning is
   what makes that attributable.
2. **Prefix-cache stability is a design constraint.** Comments like
   *"included only when ref images are present, so a sample with none produces byte-identical
   prompt output"* (`system_prompt.py:306`), reflection appended as a trailing user message
   *"to preserve KV cache prefix"* (`reflection_prompt.py:1-5`), and sticky session→endpoint
   hashing (`client.py:330-336`). At 40 GPU-hours per full-split run, this matters to you.

Content-level items worth borrowing, in priority order:

- **The coordinate-system section** (`prompt_common.py:179-226`) opens by naming the failure
  mode rather than the rule: *"Every spatial question implicitly assumes a coordinate
  system, but almost never says which one. Your first job … is to resolve this ambiguity."*
  Then four spaces (pixel / camera / world / object-perspective) with the pixel-space warning
  spelled out. Your c5 orientation category is **21.2%, below chance** — that is the
  object-perspective/camera-perspective confusion, and your facing-axis recipe
  (`dsi_bench_system.md:186-212`) gives the formula without ever naming the frame problem.
- **The dot-product direction recipe as policy**: *"ALWAYS use this quantitative method for
  direction questions. NEVER guess direction from 2D image appearance."* Your equivalent
  exists but is not marked as mandatory.
- **`robust_computation`** (`prompt_common.py:320-328`): median over mean, sanity-check
  magnitudes, *"When margins are small relative to measurement noise, acknowledge low
  confidence."* This is your Trace-1 failure (facing SNR 0.78, displacement SNR 0.09,
  reported to four decimals) written as a rule.
- **BEV legend documentation** — the prompt explains how to *read* an image the model will
  later be shown (red→blue temporal gradient, arrowhead, red start dot). If you ever render
  trajectories, teach the legend in the prompt.

---

## 6. Verification: the evaluator-optimizer pattern, applied to the terminal action

SpatialClaw gates `ReturnAnswer` behind a reflection pass (`reflection_node.py:151-178`).
On rejection it nulls `final_answer`, which routes back into the loop, and issues a
**forced three-way protocol**:

> Before resubmitting, you MUST do ONE of the following:
> - Address the concern: run the suggested verification or fix the flagged issue, then resubmit.
> - Refute the concern: if you believe your answer is correct despite the rejection, explain concretely why the concern does not apply, then resubmit.
> - Acknowledge and move on: if you cannot address or refute the concern (tool failed, data too noisy), state this explicitly and submit your best answer.
> Do NOT resubmit without doing one of the above — explain your reasoning first.

The third branch is the well-designed part: it gives the loop a legitimate exit, which is
exactly what your non-terminating runs lacked. Guarded by
`min_budget_for_answer_reject=3` and `max_total_answer_attempts=5`.

The reflection criteria are a catalogue of observed pathologies — *"Dead loops: repeating
the same action that already failed"*, *"Unsupported spatial claims"*, *"Stuck on 'Cannot
determine'"*, *"Priming in `vlm.ask_with_thinking` queries"*, *"Manual coordinate
guessing"* — plus an explicit **"What NOT to flag"** block to suppress an over-eager critic.

**Two cautions before you adopt this.**

1. `enable_reflection` defaults to **`False`** (`config.py:192`). NVIDIA shipped it off.
   Treat it as unvalidated.
2. It is the same model with the same full context — self-critique, not an independent
   judge, so it inherits every bias in the transcript it is judging.

**What to take instead:** the *cheap deterministic* subset of those criteria, enforced in
the host, no extra generation. Your two worst loops are trivially detectable:

- **Duplicate action.** `CameraBench_c1_…_95469228.json` emits byte-identical responses at
  steps 2, 3, 4… — greedy decoding over unchanged context is a fixed point, so this is
  guaranteed, not stochastic. Hash `(action, args)`; on repeat, refuse with a message that
  *changes the context* (which is what breaks the fixed point).
- **`_refuse_hopeless_query` is defeated by partial visibility.** `dsi_bench_run.py:190-195`
  marks a query blind only if *no* target was visible; with `math_visibility == [True,
  False]` the guard never fires, and that trace burned 14 steps twice.

Also fix the deadline contradiction (`_step_notice` at remaining-2 promises a step that
`HARD_DEADLINE_STEPS` then refuses — 5+ rejections in this run), and stop charging
rejections against the step budget; SpatialClaw counts **consecutive** failures and resets
on success (`llm_step_node.py:135`), and explicitly does not charge transient infra errors.

---

## 7. What NOT to copy

- **No constrained decoding.** Their four-tier reflection parser falls back to
  `if "concern" in text.lower()` — and the reflection prompt itself contains the string
  *"If you spot a concern"*. A model echoing its instructions gets its answer rejected. You
  can do better with `StoppingCriteria` today.
- **Unbounded history.** `messages` is never windowed or summarized; the only shrink path is
  retroactive error condensation via `RemoveMessage` (`feedback_node.py:161-179`). That one
  *is* worth stealing — it rewrites a failed step down to `**Reasoning**: [errored —
  condensed]` plus code-up-to-the-error-line marked `# <-- ERROR`.
- **Stale docs as a category.** `state.py:70` says *"text-only, NO images ever"*; images are
  injected at `feedback_node.py:68-92`. Prompt says 8 forbidden constructs, `safety.py`
  blocks ~28. A dead `tools.EasyOCR` description sits in the tree, fully written, wired to
  nothing.
- **Two index spaces with similar names.** `prompt_frame_idx` is window-local while
  everything else is absolute; `render_bev(ref_frame=)` is documented absolute but silently
  falls back to local (`per_frame_types.py:679-682`) — a wrong value yields a plausible wrong
  plot instead of an error. Anti-poka-yoke. They pay for it with ~8 `# CRITICAL:` shouts.
- **ReAct-mode representation mismatch.** The model emits JSON, the host transpiles to
  Python, and the *Python* is echoed into context (`llm_step_node.py:113-121`) — it never
  sees what it actually wrote.

---

## 8. Recommended sequence

Ordered by (evidence strength × cost). Steps 1–3 are prerequisites for any further prompt
experiment: until the transcript is authoritative, prompt A/Bs measure noise.

| # | Change | Grounded in | Cost |
|---|---|---|---|
| 1 | Stop generation at first complete action; append only the executed action to history | §1 — 103/120 rejections | ~1 day |
| 2 | Rendered tool-result summary; drop `sampled_to_original`; put `bindings` shape and noise floor in the observation | §3 — 95 invalid `python_math`, 2/10 std usage | ~1 day |
| 3 | Host-side duplicate-action guard; fix `_refuse_hopeless_query` partial-visibility hole; fix deadline contradiction; don't charge rejections to budget | §6 — 3 questions × 14 steps × 2 attempts | ~1 day |
| 4 | **Re-run the 25-q pilot.** No prompt changes. | isolates orchestration from prompting | 1 GPU-hr |
| 5 | `locate_target` as an isolated-session action; `query_d4rt` takes `grounding_id` | §2 — IoU 0.013→0.263, 54% invisible | ~3 days |
| 6 | Invisible-target recovery menu | §4 | ~0.5 day |
| 7 | Section the system prompt with an ablation hook | §5 — attribution | ~1 day |
| 8 | Then, and only then, coordinate-system + robust-computation sections | §5 — c5 at 21.2%, c2/c3 at chance | ~1 day |

**Statistical note.** Every comparison to date rests on n=25, where the 40% vs. 52% gap is
not distinguishable from chance (McNemar p=0.45) and 7/25 is consistent with guessing
(p=0.44). The full split has no tool-free control at all. Before step 4, stand up a
**paired baseline on ≥200 questions**; otherwise steps 1–3 will produce a number you cannot
interpret. Also note `results/dsi_bench_full/` is untracked and its judge lives only on
`feat/dsi-bench-full-run` — the 33.4% headline is currently not reproducible from any
branch, and the judge model id (`gpt-5.6-luna`) is not externally resolvable.

---

## 9. One honest structural difference

SpatialClaw gives the model a **general-purpose kernel** — arbitrary numpy, matplotlib,
scipy, persistent variables, images of its own intermediate results. The D4RT agent gives it
**three actions and a 32-statement restricted AST evaluator**. That is not obviously the
wrong call: your `_RestrictedMathEvaluator` makes the agent's arithmetic auditable and
reproducible, which is worth real accuracy for a benchmark harness, and it is why your
`python_math` outputs can be provenance-checked against the final answer at all.

But it means every capability the model needs must be anticipated by you, and the failure
mode of an unanticipated need is a rejection, not an improvisation. Their c1-equivalent
freedom is why they can inspect a BEV render and self-correct; your 43.6% on c1 shows the
measurement channel is genuinely working when grounding holds, and the c2/c3 collapse
(23.9%, 24.8%, last in field) is where a missing capability — reliable camera egomotion —
has no fallback. Worth a decision, not a drift: either add an egomotion action, or accept
c2/c3 as out of scope and say so.
