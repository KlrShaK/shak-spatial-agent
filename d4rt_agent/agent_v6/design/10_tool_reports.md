# 10 — What a tool returns

Theme 1 of the discussion. It covers tracker points T1, P3 and P9. Status: **settled** on
2026-10-17 (decision log #1–3 and #6–9).

## The decision

Every V6 measurement comes back as one report in three parts, in this order:

1. **Headline, in natural language.** It commits to direction words ("moves FORWARD", "gets
   CLOSER", "turns LEFT"). Each word sits next to the size of the motion and the noise floor it was
   judged against. The **majority motion comes first**. Then, for completeness, the **minority
   motions with when they happen**: a still start, a late turn, a brief reversal.
2. **Phase table.** One row per phase, where a phase is a run of the same direction, in time order.
   It is built from the same values as the headline.
3. **Reading notes.** The rules needed to read this report: what the axes mean ("forward/right are
   the runner's own axes, not the screen's"), what "still" means here, and what the report cannot
   tell you. The **first report from a tool in a question carries the full contract** (V4-style,
   e.g. the orbit sign rule for the camera). Later reports from the same tool carry one or two
   short lines.

**Reports never mention the answer options.** The model sees the options **from the planning step
onward**, as candidate choices. When it reads a report, it matches its reading to the options and
picks the best match or eliminates the rest. Example: the measurement reads "still, then forward",
and the only fitting option says "forward", so "forward" is chosen. The host's thresholds therefore
describe the motion but never select an option.

Tools measure over the **whole clip by default**. A window is used only when the question names a
time ("in the first 2 seconds"), and the host derives it from the question. The model never
chooses timestamps to sample a quantity that changes over time; the report already contains the
profile.

**Why.** This is V4's format: a prose summary, a table, a reading contract. V4 won on DSI with it.
V5 led with an axis table, star marks and caveats, and its descriptive label came last. Committing
to a direction word moves interpretation to the host, which computes it from numbers. Showing the
size and the noise floor next to the word keeps it checkable. Putting majority first and minority
after resolves the V5 conflict between "describe in time order" and "name the dominant motion":
the headline states dominance, and the minority clause and the table keep the order. Keeping
reports option-blind preserves "the model chooses" (no D7-style host decision). Options at planning
let the model see quantities named only in options (414 of 1,769 DSI questions mention rotation
or orbit only there).

## Example

```
subject_track — the runner, in its own starting frame (forward = where it faced at 0.9 s)
Moves FORWARD most of the clip: net 1.73 forward, 0.41 right (noise 0.05).
Still for the first 0.3 s; turns 3° right only in the last 0.3 s.
time        direction   moved   turn
0.0–0.3 s   still       0.05    –
0.3–2.8 s   forward     1.48    –
2.8–3.1 s   forward     0.11    +3° right
Reading: forward/right are the runner's own axes, not the screen's. "still" means the motion
in that phase stayed below the noise floor.
```

## Implementation notes

- **One shared report record and renderer for every tool.** The record holds a title line (entity,
  frame, reference), headline, phase rows, reading notes (full and short variants), a status (`ok`,
  `partial`, `unavailable`) with a reason, and raw data. The prompt shows only the rendered text;
  raw series stay in the saved trace. The host tracks which tools have already shown their full
  contract in the current question.
- **Phases.** Label fine equal-time steps (for example 20 over the clip) with direction words
  computed against the noise floor. Merge consecutive steps with the same label. Absorb a phase
  shorter than a minimum duration into its neighbours unless its motion clears the noise floor
  several times over. Cap the table at about six rows. The minimum duration and the multiple are
  geometric choices fixed up front, not fitted on accuracy.
- **Headline from the phases.** Majority = the direction covering the most time or distance, stated
  first, with net components and the noise floor. Minority = every other phase above the floor, in
  time order, with its time span. Start from V4's `traj3d_buckets.segment_report`, `_segment_prose`
  and `rank_components`.
- **Units.** Scene-relative units, with the note "comparable within this clip only". Metres only
  when a metric scale exists, with its source named (Point4D × DA3).
- **Time labels.** Seconds when the clip's frame rate is known. Otherwise labels that match the
  frames the model was shown. Never the generator-frame integers (P4).
- **Unavailable reports.** One line naming what failed and why, with no numbers. What happens next
  belongs to Theme 4 (doctor and retries) and Theme 5 (answering).
- **Reuse.** V4's table renderer `agent_v4/orchestrator._render_metrics` and its four contracts in
  `agent_v4/prompts/contract_*.md` as the first full contracts; V5's time-correct numerics in
  `agent_v5/measure.py` (actual timestamps, unwrapped angles, roll included), but not its rendering.

## To check when implemented

- Majority/minority headlines on a small synthetic bank: early-left-then-right, still-then-forward,
  out-and-back. Every phase above the floor must appear, and the majority must match the table.
- Phase tables never exceed the row cap, and a jittery but static track yields a single "still" phase.
