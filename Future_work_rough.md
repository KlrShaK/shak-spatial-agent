❯ Make a new worktree

  In the first run of QuickCheck set evaluation for AgentV6 we didn't get as good results as we had hoped for I think the planning pipeline is contraining the model to output plans and then it outputs shit plans and then fails.

  Also the tool outputs are very messy and shitty compared to V4. The natural language summary on top of tool output is too noisy instead of highlighting majority motion it highlights every small motion that too with every timestamp. Instead it should be clean and human readable. It should be less than or ~50 words

  Don't think of the agent as superhuman consider it a 10 year old, your prompts should be readable and understandable enough that a 10 year old can understand them and answer.

  We dont give any visual frames to agent in beginning.

  Build/Refactor Agent_v5 but limit the toolset to

  | Tool | Arguments | Reports (whole clip unless the question names a time) |
  |---|---|---|
  | `detect_all_entities` | called by the model when it wants the full inventory (#30) | One Qwen call lists the nouns in question, options and caption; then the `get_segmentation_mask` table for every noun, plus instance counts per category and segmented frames with instance labels (#40) |
  | `track_camera` | — | The camera's own motion in its start view (gravity-levelled): translation and pan/tilt/roll per phase, tempo, jitter |
  | `track_entity` | `main`, `others=[]`, `frame_of_reference` = `world` / `camera` / `subject` | **world:** paths of `main` (and `others`) in the start view, plus each one's own turning. **camera:** as the moving camera sees them: towards/away, left/right in the view, distance to the camera. **subject:** `main`'s own motion in its *starting* facing, plus where each of `others` is relative to `main` in main's facing *at each moment* (#26). If `others` is empty, that relative part is about the camera; if `others` is given, the camera is left out (#27). |
  | `track_orientation` | `main`, `frame_of_reference` = `camera` / `own` / an entity | **camera:** which side `main` shows the camera over time. **own:** `main`'s turning relative to its starting facing, with camera motion removed. **entity:** `main`'s facing relative to that entity. |
  | `track_multiple_entities` | `category`, `frame_of_reference` = `world` / `camera` / `own` (#42) | Every instance of the category: direction of each, spread over time, counts by direction |
  | `track_camera_zoom` | — | Focal ratio and forward motion |
  | `check_scene_revisit` | — | How much of the start view is seen again at the end |
  | `get_segmentation_mask` | `entities` (required), `times` (seconds, optional) | Runs on the whole video (#39). A table with one row per labelled instance: visible spans, size as a share of the image in phases (`6% → 14% → 9%`), and when it touches the image edge (#40). With `times`, also up to 8 frames at those times with the masks; without, the table only. Entity names come from the planning step, or from `detect_all_entities` first. |
  | `get_frames` | `times` (up to 4, seconds) **or** `range` (start, end in seconds) | 4 plain frames: at the given times, or sampled uniformly over the range, each labelled with its time (#41) |
  | `ask_video` | an open question in words, e.g. "At which time does the skier jump?"; for visual questions it may carry the benchmark question and options (#52) | A separate VLM agent answers from the clip, with times in seconds (#44, #55) |
  | `choose_segmentation_candidate` | the colour of the intended instance | Available only right after a call that asked the agent to choose an entity: one image with each SAM3 candidate in its own colour; the agent picks one (#48, #55) |
  | VLM only | — | The counting decision; anything no tool reports |
  and the families could be (Kinds are removed); the concept of family here is much looser, here the family selection is just for agent to get more specific info and family can be changed at any time (we will show a brief description about families)
  VIDEO
   │
   ├─────────────────────────────────────────────┐
   │                                             │
   ▼                                             ▼
  ENTITY PERCEPTION                         CAMERA PERCEPTION
   │                                             │
   ├─ detect entities                            ├─ ego-motion
   ├─ segment instances                          ├─ zoom/dolly

   │
   ▼
  SUBJECT MOTION
   │
   ├─ position / trajectories
   ├─ orientation
   ├─ subject-object relations
   └─ multi-instance motion

                   +

  GENERAL VISUAL INSPECTION
   ├─ get frames
   ├─ ask VLM
   └─ VLM-only judgments

  My idea is to seprate reasoning and measurements, we leave reasoning to VLM and measurs which they are good at.

  I introduce another new tool called save_evidence(), which the agent can use to store to answering a question
  TOOL CALL
     │
     ▼
  TOOL RESULT
     │
     ▼
  Agent asks internally:
  "Does this result contain evidence relevant to answering
  the benchmark question?"
     │
     ├── No ───────────────► continue
     │
     └── Yes
          │
          ▼
     save_evidence(...)
          │
          ▼
     continue investigation
  We should introduce also some kind of Question Decomposition
  GOAL:
  G0 = determine whether camera_pan_left happens before skier_jump

  DEPENDENCIES:

  G1 = determine time of skier_jump
  G2 = determine onset time of camera_pan_left

  G0 depends on [G1, G2]
  dependencies can be empty if the question is simple enough; The initial agent is given @d4rt_agent/agent_v6/design/11_toolset.md lines #16-31 and asked based on that to decompose questions
