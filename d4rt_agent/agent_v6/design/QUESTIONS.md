# AgentV6 implementation questions

## Provisional choices required by the plan

1. **Backbone:** Point4D by default; D4RT selectable with `--backbone`. Both use the existing V5 backbone interface. This is provisional and reversible by configuration.
2. **Quick-check set:** implement deterministic, parameterized selection using the proposed pool sizes/counts; do not generate the real selection until Shak approves it.

## Clarifications during implementation

- Dry run writes only one answer stub per selected question, with `status=dry_run` and no prediction. It never opens clips or loads a model. Real catalog checks run in a CPU SLURM job; unit fixtures are small and GPU-free.

- **Report example inconsistency (provisional fixture choice):** design 10's net displacement and phase distances cannot all describe the same synthetic trajectory. Preserve its literal rendering as an explicit record fixture; check computed geometry independently. No numbered decision changed.
- **Phase geometry:** a Euclidean displacement noise floor and source-sample-aware simplification retain coherent slow motion and brief strong reversals. Short weak phases may be absorbed as allowed by design 10; reading notes state that suppression explicitly. More than six strong phases are packed into six display rows without discarding their labels or times.
- **Instance identities (provisional geometric choice):** refresh matching requires mask IoU >0.5; labels are never recycled for different masks. SAM session IDs are not persistent identities.
- **Unknown fps:** reports reuse labels of displayed frames; isolated video helpers use the same eight keyframes as the main agent, since seconds cannot be inferred. Numeric time windows are unavailable without a frame rate.
- **Monitoring:** no Monitor tool is available in this session. The CPU-only validation job is checked with SLURM status/output commands. Initial sandbox scheduler access failed; the authorized submission succeeded with scheduler access. Pending submissions were replaced with a five-minute, single-CPU request directly on normal.4h; latest job 17404184. No GPU resources were requested.
- **Time windows (provisional parser scope):** the host recognizes explicit first/last N seconds and between/from X to Y seconds in the question stem. No tool schema exposes generator-frame windows or lets the model silently choose measurement samples.

- **Plan completion (provisional protocol detail):** the model may include `completed_plan_items` in its next call/answer envelope. Auxiliary calls and successful reports do not automatically complete planned quantities. Plan revisions reset completion; this keeps execution-rate metrics tied to explicit model declarations.
- **Group tables:** per-instance rows retain all grounded members; the temporal summary is capped at six rows. Collapsing instance rows to six would conceal group members.
- **Dyn levels:** the existing `TASK_TO_LEVEL` mapping calls itself a hypothesis. Report actual task labels and hypothetical levels separately until a verified per-question level manifest is supplied with `--metadata-file`. Never present the hypothesis as an official paper level.
- **Frozen-run external paths:** `--third-party-dir` defaults to `SHAK_THIRD_PARTY` or the existing repository's pinned tools. Ignored metadata artifacts have a lazy shared-repository fallback. These external inputs are configurable; source snapshots do not copy weights, datasets or submodules.
- **Video runtime validation:** synthetic processor/wrapper tests and installed source inspection cover sampling, timestamps and token budgeting. A slow CPU processor import smoke was stopped; actual Qwen video inference and backbone compatibility are reserved for M9 GPU validation.
