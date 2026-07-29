• # Generic single-action orchestration design

  ## 1. Goal

  The agent should operate as a real tool-using loop:

  1. Think about the current evidence.
  2. Submit exactly one action.
  3. Stop and wait.
  4. Receive the real host result.
  5. Update its reasoning.
  6. Submit the next action.
  7. Eventually call final_answer.

  The model may keep its thinking aloud in conversation history. What must be prevented is generating speculative future actions or simulated tool results before the host has responded.

  This orchestration should remain generic. It should work for D4RT, calculators, the proposed Qwen grounding tool, or future tools without embedding DSI-specific recovery policies.

  ———

  ## 2. Current failure mechanism

  The current host already executes only the first action it finds. The problem is that it appends the model’s entire generated response to history before executing that action.

  Qwen often generates something like:

  I should measure the runner first.

  {"action":"query_d4rt", ...}

  Tool result d4rt_1: ...

  Now I will calculate the displacement.

  {"action":"python_math", ...}

  The answer is B.

  {"action":"final_answer", ...}

  The host executes only query_d4rt, but the entire imagined transcript remains in history. On the next turn, Qwen believes its fabricated tool result and later actions actually happened.

  This caused:

  - References to nonexistent d4rt_2 and math_1.
  - Repetition of the same rejected calculation.
  - Premature final answers based on fabricated numbers.
  - Multiple actions in 37.5% of responses.
  - Four questions exhausting strict and relaxed attempts.

  The key distinction is:

  - Reasoning before the first action should remain.
  - Everything after the first action must not enter effective history.

  ———

  ## 3. Core decisions

  ### Decision 1: one assistant turn equals one action

  Every assistant response has this structure:

  Optional thinking aloud in plain text.

  {"action":"ONE_ACTION","arguments":{...}}

  The JSON action must be the end of the response.

  The agent must then wait for the host.

  ### Decision 2: preserve thinking aloud

  The reasoning before the submitted action remains in conversation history. It is useful for continuity, planning, and debugging.

  We are not replacing the response with canonical JSON alone.

  ### Decision 3: exclude speculative continuation

  Anything after the first complete action is:

  - Saved in the raw diagnostic trace.
  - Excluded from model-visible conversation history.
  - Never treated as a real tool result or action.

  ### Decision 4: the host owns evidence state

  Only successfully executed host actions create evidence identifiers.

  The model may reference IDs, but it cannot create them. After every action, the host provides an authoritative ledger listing exactly what exists.

  ### Decision 5: prompt guidance and host enforcement are both required

  The prompt teaches the intended workflow. The host guarantees it.

  Prompt-only enforcement already failed, so correctness cannot depend on Qwen following the instruction voluntarily.

  ### Decision 6: use separate-turn worked examples

  The system prompt should include complete workflows so Qwen learns how tools compose, including final_answer. However, examples must be formatted as clearly separated assistant and host turns.

  Host-side stopping ensures Qwen cannot reproduce several example turns in one live generation.

  ———

  ## 4. Generic interaction protocol

  A short protocol should be prepended to every tool-using system prompt:

  ## Tool interaction protocol

  Each assistant response is exactly one turn.

  During a turn:

  1. You may think aloud in plain text.
  2. Choose exactly one action.
  3. End your response with exactly one JSON action.
  4. Stop immediately after the action's closing brace.
  5. Wait for the host's response before deciding what to do next.

  Never:

  - Emit two actions in one response.
  - Simulate or predict a tool result.
  - Write "Tool result".
  - Invent an evidence identifier.
  - Continue reasoning after the action.
  - Emit a future action before the current action has returned.
  - Write JSON objects in your reasoning.

  The first action JSON is the action submitted to the host.
  Only the host creates tool results and evidence identifiers.

  This should be host-owned and generic, rather than copied independently into every benchmark prompt.

  A possible implementation is a shared prompt fragment:

  d4rt_agent/prompts/tool_turn_protocol.md

  The orchestrator prepends it to the task-specific system prompt.

  ———

  ## 5. Worked-example format

  Complete workflows remain valuable because the model must learn how querying, calculation, and final answering fit together.

  They should look like this:

  ASSISTANT TURN 1

  I need the runner's position at the beginning and end in one fixed
  camera frame. I will measure that first.

  {"action":"query_d4rt","arguments":{
    "label":"runner",
    "bbox_2d_1000":[447,350,475,440],
    "t_src":0,
    "t_tgt":[0,31],
    "t_cam":0,
    "justification":"Measure endpoint positions in one fixed frame."
  }}

  --- END ASSISTANT TURN; WAIT FOR HOST ---

  HOST TURN

  Tool result d4rt_1:
  {
    "predictions":[
      {"benchmark_aligned_xyz_m":[-0.4,-0.6,5.5]},
      {"benchmark_aligned_xyz_m":[-0.4,-0.6,3.8]}
    ]
  }

  HOST EVIDENCE STATE:
  - d4rt_1: query_d4rt

  --- END HOST TURN ---

  ASSISTANT TURN 2

  I now have two endpoint positions. I will calculate the displacement
  and range change.

  {"action":"python_math","arguments":{
    "bindings":{
      "start":{
        "evidence_id":"d4rt_1",
        "path":["predictions",0,"benchmark_aligned_xyz_m"]
      },
      "end":{
        "evidence_id":"d4rt_1",
        "path":["predictions",1,"benchmark_aligned_xyz_m"]
      }
    },
    "code":"dz = end[2] - start[2]\nstart_range = norm(start)\nend_range = norm(end)",
    "justification":"Calculate the runner's depth and range change."
  }}

  --- END ASSISTANT TURN; WAIT FOR HOST ---

  HOST TURN

  Tool result math_1:
  {
    "outputs":{
      "dz":-1.7,
      "start_range":5.6,
      "end_range":3.9
    }
  }

  HOST EVIDENCE STATE:
  - d4rt_1: query_d4rt
  - math_1: python_math

  --- END HOST TURN ---

  ASSISTANT TURN 3

  The large negative depth change supports forward motion.

  {"action":"final_answer","arguments":{
    "kind":"text",
    "text":"B: Moving forward. Depth decreased substantially from the start to the end.",
    "evidence_ids":["d4rt_1","math_1"],
    "limitations":"The conclusion uses one tracked point on the runner."
  }}

  --- END ASSISTANT TURN ---

  Important example rules:

  - Only the host writes HOST TURN.
  - Every assistant example ends immediately after one action.
  - At least one example must reach final_answer.
  - Examples should cover tool composition, not only isolated syntax.
  - Host results should be short and clearly illustrative.

  ———

  ## 6. Action extraction

  The action extractor currently returns only the parsed action. It should return the exact character span as well:

  @dataclass
  class ExtractedAction:
      value: dict[str, Any]
      start: int
      end: int

  Suggested interface:

  def extract_first_action(text: str) -> ExtractedAction:
      ...

  It should:

  1. Search for possible JSON object starts.
  2. Use json.JSONDecoder.raw_decode.
  3. Find the first complete object containing action or name.
  4. Return the parsed object and its exact start/end positions.

  Conceptually:

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

  raise ValueError("No action JSON found")

  Using the JSON decoder handles:

  - Nested dictionaries.
  - Arrays.
  - Escaped quotes.
  - Braces inside strings.

  The system prompt should still prohibit JSON inside reasoning so the first action object is unambiguous.

  ———

  ## 7. Corrected orchestration order

  The current problematic order is:

  generate
  append complete raw response to history
  extract first action
  execute
  append result

  The corrected order is:

  generate
  extract first action and its boundary
  save complete raw response in trace
  append only reasoning + first action to history
  validate action
  execute action
  register evidence if successful
  append real tool result
  append evidence ledger
  generate next assistant turn

  Pseudocode:

  raw = qwen.generate(messages)

  attempt = {
      "step": step,
      "kind": "model_action",
      "raw_qwen_response": raw,
  }

  try:
      extracted = extract_first_action(raw)

      effective_response = raw[:extracted.end].strip()

      messages.append({
          "role": "assistant",
          "content": [{
              "type": "text",
              "text": effective_response,
          }],
      })

      action_name, arguments = validate_action(extracted.value)

      result = execute_action(action_name, arguments)

      evidence_id = register_successful_result(
          action_name,
          result,
      )

      trace.append({
          **attempt,
          "effective_response": effective_response,
          "discarded_suffix": raw[extracted.end:],
          "status": "ok",
          "call_id": evidence_id,
          "result": result,
      })

      messages.append({
          "role": "user",
          "content": [
              {
                  "type": "text",
                  "text": format_tool_result(evidence_id, result),
              },
              {
                  "type": "text",
                  "text": format_evidence_ledger(evidence),
              },
          ],
      })

  except ActionError as error:
      ...

  The full raw response remains available for debugging, while effective_response records exactly what entered conversation history.

  ———

  ## 8. Rejected-action handling

  ### Parsed but invalid action

  If JSON is found but fails schema or evidence validation:

  1. Preserve the reasoning and submitted action through action_end.
  2. Exclude anything after it.
  3. Do not create an evidence ID.
  4. Append the rejection.
  5. Append the evidence ledger.
  6. Start a new assistant turn.

  Example:

  Action rejected:
  Evidence ID "d4rt_2" does not exist.

  HOST EVIDENCE STATE — authoritative

  Available evidence:
  - d4rt_1: query_d4rt

  Last action: rejected
  New evidence created: none

  Rejected actions never create evidence identifiers.
  Use an existing identifier or submit the missing tool call as your next action.

  ### No parseable action

  If no action object exists:

  - Save the complete response in the trace.
  - Preserve its plain-text reasoning in history.
  - Append a concise rejection:

  No action JSON was found. Your next response may contain reasoning,
  but it must end with exactly one action JSON.

  No evidence ID is created.

  ### Tool execution failure

  A schema-valid action may fail during execution. The result should be recorded as a failed attempt, not successful evidence:

  Last action: execution failed
  New evidence created: none
  Reason: ...

  A separate diagnostic attempt ID may be useful internally, but it must not appear in the usable evidence list.

  ### Valid but null result

  A successfully executed tool call that returns null or invisible measurements is still real evidence. It receives an evidence ID because the tool actually ran.

  The ledger records existence, not usefulness:

  - d4rt_2: query_d4rt, status=ok

  Deciding whether that result is useful remains the reasoning agent’s responsibility for now.

  ———

  ## 9. Evidence ledger

  The evidence ledger is a compact, host-generated view of authoritative state.

  ### Successful action

  HOST EVIDENCE STATE — authoritative

  Available evidence:
  - grounding_1: ground_with_qwen
  - d4rt_1: query_d4rt

  Last action: accepted
  New evidence created: d4rt_1

  Only identifiers listed under Available evidence exist.

  ### Rejected action

  HOST EVIDENCE STATE — authoritative

  Available evidence:
  - grounding_1: ground_with_qwen
  - d4rt_1: query_d4rt

  Last action: rejected
  New evidence created: none
  Reason: math_1 does not exist

  Rejected actions do not create evidence identifiers.

  ### Design rules

  - Generated entirely from host state.
  - Appended after every success or rejection.
  - Contains IDs and tool types, not the entire evidence payload.
  - Kept short to avoid context growth.
  - Does not contain benchmark-specific advice.
  - Makes clear that the host is authoritative.
  - Evidence counters advance only after successful execution.

  The ledger can be represented internally as:

  @dataclass
  class EvidenceEntry:
      evidence_id: str
      tool_name: str
      status: Literal["ok"]
      result: dict[str, Any]
      created_at_step: int

  The model-facing ledger uses only the metadata. Full results remain in the evidence registry and trace.

  ———

  ## 10. More instructive generic evidence errors

  Unknown-evidence errors should include actual available IDs:

  def unknown_evidence_error(requested_id, evidence):
      available = sorted(evidence)
      return (
          f"Evidence ID {requested_id!r} does not exist. "
          f"Available evidence IDs: {available}. "
          "Rejected actions create no evidence. "
          "Use one of the available IDs or submit the missing tool call "
          "as this turn's single action."
      )

  This is generic and applies to all tools.

  The error should avoid saying only:

  binding cites unknown evidence

  That identifies the error but does not correct the model’s false internal state.

  ———

  ## 11. Generation stopping

  There are two implementation stages.

  ### Stage A: history-boundary enforcement

  Allow Qwen to generate normally, then:

  - Find the first action.
  - Preserve only raw[:action_end] in conversation history.
  - Save the entire raw generation in the trace.

  This is the minimum correctness fix and should be implemented first.

  It prevents phantom future actions from affecting later turns even if Qwen continues generating them internally.

  ### Stage B: token-level stopping (OPTIONAL- Don't implement yet)

  Add a Transformers stopping criterion so generation itself stops when the first action closes.

  The stopping logic tracks:

  - Whether action JSON has started.
  - JSON brace depth.
  - Whether the current position is inside a string.
  - Escape characters.
  - Completion of the top-level action object.

  When the first valid action object closes, generation terminates.

  Benefits:

  - Less inference time.
  - Smaller raw responses.
  - Less opportunity for Qwen to form a multi-action continuation.
  - Exact correspondence between generated response and executed turn.

  Stage B is an optimization and additional safeguard. Stage A establishes correctness.

  ———

  ## 12. Trace structure

  Each model turn should retain both raw and effective responses:

  {
    "step": 1,
    "kind": "model_action",
    "raw_qwen_response": "reasoning + action + imagined future actions",
    "effective_response": "reasoning + first action",
    "discarded_suffix": "imagined future actions",
    "parsed_action": {
      "action": "query_d4rt",
      "arguments": {}
    },
    "status": "ok",
    "call_id": "d4rt_1",
    "result": {}
  }

  This allows us to measure:

  - How frequently Qwen attempted multiple actions.
  - Whether host enforcement removed them correctly.
  - Whether prompt changes reduce discarded suffixes.
  - What information the agent actually saw in later turns.

  The report should render effective_response as the actual conversation while optionally exposing raw_qwen_response under diagnostic details.

  ———

  ## 13. Integration with the proposed grounding tool

  The same generic protocol naturally supports the separate Qwen grounding tool.

  ### Turn 1: request grounding

  ASSISTANT TURN

  I need a reliable box around the runner before tracking it.

  {"action":"ground_with_qwen","arguments":{
    "mode":"bbox",
    "t_src":0,
    "request":"runner"
  }}

  ### Host response

  Tool result grounding_1:
  {
    "grounding_id":"grounding_1",
    "bbox_2d_1000":[447,350,475,440]
  }

  HOST EVIDENCE STATE:
  - grounding_1: ground_with_qwen

  ### Turn 2: query D4RT

  ASSISTANT TURN

  The runner is now grounded. I will measure its endpoint positions.

  {"action":"query_d4rt","arguments":{
    "grounding_id":"grounding_1",
    "t_tgt":[0,31],
    "t_cam":0
  }}

  ### Turn 3: calculate

  {"action":"python_math", ...}

  ### Turn 4: answer

  {"action":"final_answer", ...}

  This demonstrates why generic single-action orchestration should be fixed before adding the grounder: every new tool can rely on the same waiting, evidence, and provenance behavior.

  ———

  ## 14. Tests

  ### Action-boundary unit tests

  - Preserve all reasoning before the first action.
  - Retain exactly one complete action.
  - Exclude a second action.
  - Exclude fabricated Tool result text after the first action.
  - Correctly handle nested objects and arrays.
  - Correctly handle escaped quotes and braces inside strings.
  - Save the complete raw response in the trace.
  - Record the discarded suffix.

  ### Evidence-ledger tests

  - Successful actions create one evidence ID.
  - Rejected actions create no evidence.
  - Execution failures create no usable evidence.
  - Existing evidence remains after a rejection.
  - Ledger IDs exactly match the evidence registry.
  - Unknown-evidence messages list available IDs.
  - Counters do not advance on rejection.

  ### Conversation tests

  Using a scripted fake Qwen:

  1. Qwen emits reasoning, a valid query, a fake result, and a calculator call.
  2. Only the query enters effective history.
  3. The host returns the real result and ledger.
  4. Qwen’s second turn receives only authoritative evidence.
  5. The second action is executed.
  6. A later final_answer terminates the loop.

  ### Prompt tests

  - The interaction protocol explicitly requires one action.
  - It forbids simulated tool results.
  - It says to wait for the host.
  - Worked examples contain clear assistant/host turn boundaries.
  - At least one complete example ends with final_answer.

  ### Pathological replay cases

  Use the latest traces:

  - #3: phantom d4rt_2.
  - #17: repeated calculation and final answer using nonexistent evidence.
  - #22: valid action followed by an invalid future action.

  Success means:

  - One effective action per model turn.
  - No simulated host results in effective history.
  - No nonexistent evidence references.
  - The next action occurs only after the real host response.

  ———

  ## 15. Implementation sequence

  1. Add a shared generic tool-turn protocol.
  2. Reformat worked examples into explicit assistant/host turns.
  3. Extend action extraction to return character boundaries.
  4. Move history insertion until after action extraction.
  5. Store full raw responses only in traces.
  6. Insert only reasoning plus the first action into effective history.
  7. Add the authoritative evidence ledger.
  8. Improve generic unknown-evidence errors.
  9. Add unit and scripted-conversation tests.
  10. Replay the three pathological questions.
  11. Add token-level stopping criteria.
  12. Only then integrate the Qwen grounding tool.
  13. Finally rerun the fixed 25-question subset.

  ———

  ## 16. Explicitly deferred work

  To keep the agent generic, the following policies are deferred:

  - DSI-specific partial-visibility handling.
  - Automatic duplicate D4RT-query rejection.
  - Task-specific forced-final behavior.
  - Changes to relaxed retry policy.
  - Text-level validation of quantitative claims.
  - Special handling for camera-motion questions.
  - Benchmark-specific answer heuristics.

  Some may still prove valuable later, but the next experiment should first determine how much is fixed by clean turn boundaries and authoritative evidence state.

  ## Final design principle

  The model owns reasoning and tool choice. The host owns turn boundaries, execution, and evidence truth.

  That produces the intended contract:

  Think → submit one action → stop → receive reality → think again
