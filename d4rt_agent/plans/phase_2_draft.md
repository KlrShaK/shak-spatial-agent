## 1. Motivation and hypothesis

  The standalone grounding experiment showed that Qwen can often produce reasonable boxes when given:

  - One frame.
  - One target label.
  - One simple grounding instruction.
  - No D4RT reasoning or long conversation history.

  In contrast, asking the reasoning agent to generate coordinates inside its chain of tool use did not help overall:

  - Forced-grounding agent: 7/25
  - Previous agent: 10/25
  - Qwen-only baseline: 13/25
  - Four questions failed to terminate.
  - Only 60% of D4RT calls actually included the requested grounding declaration.

  The hypothesis is therefore that visual grounding and 4D reasoning interfere when performed in the same generation. The proposed tool separates them.

  ## 2. Conceptual architecture

  The main agent remains responsible for:

  - Understanding the DSI question.
  - Deciding which objects or scene points are useful.
  - Choosing source and target frames.
  - Interpreting D4RT trajectories.
  - Selecting the final answer.

  A separate, stateless Qwen grounding tool is responsible only for:

  - Looking at one requested frame.
  - Understanding a short natural-language grounding request.
  - Returning a box or requested 2D points.
  - Returning not_found when the request cannot be satisfied.

  This should be a logically separate Qwen context. It does not necessarily require loading a second physical copy of the checkpoint. The same loaded model could be invoked sequentially with a fresh
  grounding-only conversation, avoiding unnecessary VRAM usage.

  The grounding Qwen should not see:

  - The DSI question.
  - Previous agent reasoning.
  - D4RT measurements.
  - Other frames.
  - The answer options.

  ## 3. Tool interface

  The tool has two modes.

  ### Bounding-box mode

  Used for tracking a whole visible object:

  ground_with_qwen(
      mode="bbox",
      t_src=0,
      request="the red car closest to the camera"
  )

  Example result:

  {
    "grounding_id": "qg_1",
    "status": "ok",
    "mode": "bbox",
    "t_src": 0,
    "label": "red car closest to the camera",
    "bbox_2d_1000": [447, 350, 615, 580]
  }

  ### Points mode

  Used for specific object parts, landmarks, or background features:

  ground_with_qwen(
      mode="points",
      t_src=0,
      request="runner's chest and back",
      count=2
  )

  ground_with_qwen(
      mode="points",
      t_src=0,
      request="five static points on the distant background",
      count=5
  )

  ground_with_qwen(
      mode="points",
      t_src=0,
      request="distinct corners on the building",
      count=3
  )

  Example result:

  {
    "grounding_id": "qg_2",
    "status": "ok",
    "mode": "points",
    "t_src": 0,
    "points_2d_1000": [
      {
        "point_id": "p1",
        "xy": [312, 284],
        "description": "left corner of the upper window"
      },
      {
        "point_id": "p2",
        "xy": [571, 301],
        "description": "right corner of the roof"
      }
    ]
  }

  Coordinates follow the existing normalized 0–1000 convention.

  ## 4. Point-selection behavior

  The grounding system prompt should interpret requests according to their purpose.

  For one named object:

  - Select a point securely inside the visible surface.
  - Avoid silhouettes and uncertain boundaries.

  For multiple points on one object:

  - Select distinct and spatially separated locations.
  - Prefer textured, visually recognizable parts.
  - Avoid returning several nearly identical points.

  For named semantic parts:

  - Return one separately labelled point for each requested part.
  - Preserve the requested ordering where possible.

  For generic background points:

  - Select static, high-contrast landmarks.
  - Distribute them across different image regions.
  - Prefer building corners, window corners, poles, rocks, trunks, or rigid edges.
  - Avoid people, vehicles, animals, reflections, shadows, sky, moving foliage, and temporary objects.
  - Avoid placing all points on one object or along one line.

  If the requested target is not visible, the tool should return:

  {
    "grounding_id": "qg_3",
    "status": "not_found",
    "mode": "bbox"
  }

  For point requests, returning fewer points than requested is preferable to inventing unreliable points.

  ## 5. Connecting grounding to D4RT

  The coordinates should be visible in the agent’s conversation so it can inspect and reason about them. However, the agent should not manually copy them into the D4RT call.

  Instead, D4RT should consume the immutable grounding identifier.

  For a box:

  query_d4rt(
      grounding_id="qg_1",
      t_tgt=[0, 31],
      t_cam=0,
      point_mode="ensemble5"
  )

  The host resolves qg_1 and applies the existing ensemble5 sampling within that box.

  For explicit points:

  query_d4rt(
      grounding_id="qg_2",
      point_ids=["p1", "p2"],
      t_tgt=[0, 31],
      t_cam=0
  )

  D4RT should query exactly those points and preserve separate trajectories for each point_id.

  Using identifiers rather than copied coordinates:

  - Prevents transcription and normalization mistakes.
  - Prevents the reasoning model from silently changing a box.
  - Preserves grounding provenance.
  - Makes caching and visualization straightforward.
  - Allows subsets of points to be selected safely.
  - Keeps the experiment focused on the grounder’s actual output.

  Each grounding must be bound internally to its video and source frame. The host should reject using an identifier with another clip or t_src.

  ## 6. Caching and loop prevention

  Groundings should be cached using a canonical key such as:

  (video, t_src, mode, normalized request, count)

  Identical requests should return the existing grounding without another Qwen generation.

  Additional controls should include:

  - A small maximum number of grounding calls per question.
  - A maximum point count, perhaps 5–8 initially.
  - Rejection of identical repeated D4RT queries.
  - No automatic relaxed retry after step exhaustion or a repeated-action loop.
  - Exact schema validation for boxes and points.
  - Bounds checking for all 0–1000 coordinates.
  - Immutable grounding results once registered.

  These controls directly address the repeated 14-step calls that made the previous experiment take almost three hours.

  ## 7. Expected agent workflow

  A normal reasoning sequence would become:

  1. Determine what geometric evidence is needed.
  2. Ask the grounding tool for an object box or scene points.
  3. Inspect the returned coordinates and descriptions.
  4. Reference the grounding in a D4RT query.
  5. Calculate motion from the resulting trajectories.
  6. Answer the benchmark question.

  For example:

  Need camera motion from static scene landmarks.

  ground_with_qwen(
      mode="points",
      t_src=0,
      request="five static background landmarks",
      count=5
  )

  query_d4rt(
      grounding_id="qg_4",
      t_tgt=[0, 31],
      t_cam=0
  )

  The main agent never generates raw coordinates itself.

  ## 8. Logging and visualization

  Every grounding call should save:

  - Requested frame.
  - Natural-language request.
  - Mode and requested count.
  - Raw grounding-model response.
  - Parsed box or points.
  - Grounding identifier.
  - Whether the result came from cache.
  - An image with the returned box or labelled points overlaid.

  D4RT evidence should record which grounding and point IDs produced each trajectory. This creates an auditable chain:

  agent request → qg_4 → points p1…p5 → D4RT trajectories → final answer

  ## 9. Known limitations

  This tool isolates localization, but it will not automatically fix:

  - The agent choosing the wrong semantic object.
  - Ambiguous requests such as “the tree” when several trees are visible.
  - Fine-grained parts that are visually indistinguishable; chest/back grounding was weak in the standalone probe.
  - Incorrect forward/backward conventions.
  - Incorrect coordinate-frame selection.
  - Faulty interpretation of otherwise correct D4RT trajectories.
  - Qwen selecting visually plausible but poorly trackable background points.

  It should therefore be viewed as a grounding improvement, not a complete reasoning solution.

  ## 10. Evaluation plan

  Before integrating it into the full agent:

  1. Test points mode independently, especially background landmarks.
  2. Verify point precision, spatial distribution, static-scene suitability, and not_found behavior.
  3. Inspect saved overlays manually.

  Then rerun the same fixed 25-question subset with:

  - The original reasoning behavior.
  - The new grounding tool.
  - The same model, frames, sampling seed, D4RT settings, and step budget.
  - No manual coordinate generation.
  - No other reasoning-prompt changes.

  Compare:

  - Accuracy and completion rate.
  - Grounding quality.
  - Number of grounding and D4RT calls.
  - Duplicate calls and rejected actions.
  - D4RT visibility rates.
  - Runtime.
  - Question-level flips against both previous agent runs.

  The clean experimental question is: does isolated, tool-mediated Qwen grounding improve D4RT-agent performance when the reasoning agent cannot create or modify coordinates itself?
