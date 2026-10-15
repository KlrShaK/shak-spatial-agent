# VLM4D: the dataset and its questions

Hand analysis, not generated: `TAXONOMY_REPORT.md` holds the generated numbers and is rewritten on every
`taxonomy_report` run. This file explains them. Its companion `BUCKET_REQUIREMENTS_V5.md` goes kind by kind
through what AgentV5 needs.

Paper: Zhou, Vilesov, He et al., *VLM4D: Towards Spatiotemporal Awareness in Vision Language Models*,
ICCV 2025, arXiv 2508.02095. The PDF and the official repo, including the graded outputs of 23 models, are at
`/cluster/work/igp_psr/spanwar/VLM4D`. The offline data is at `/cluster/work/igp_psr/spanwar/datasets/VLM4D`
(`QA_local/` has the same rows as `QA/` with local video paths).

## 1. What it tests and how it was built

VLM4D asks whether a VLM can tell **which way things move and turn** in short videos. Its first figure is a
car that turns left in its own frame but moves right on screen. Every question is four-option multiple
choice.

- **Real video (1371 questions, 600 clips).** DAVIS and YouTube-VOS supply the exo-centric clips and Ego4D the
  ego-centric ones. Each clip is cut down to "the most relevant action", 1-15 s long. Humans wrote the
  questions and the correct answers. GPT-4o wrote the three distractors, and three rounds of cross-checking
  followed.
- **Synthetic video (445 questions, 400 clips).** Cosmos generated the clips with bounding-box guidance, so
  the object really follows the prompted direction. GPT-4o wrote the questions from the generation prompt.
  Almost all follow one template, "What direction is the ⟨object⟩ moving?", with the options left / right /
  not moving / "no ⟨object⟩ there". 10% are false positives that ask about an object that is not in the clip.

| view | source | questions | clips | clip length (median, range) | fps | resolution |
|---|---|---|---|---|---|---|
| exo | DAVIS | 795 | 210 | 3.1 s (1.0-5.3) | 24 | 854x480 |
| exo | YouTube-VOS | 127 | 90 | 3.3 s (1.2-6.0) | 30 | 1280x720 |
| ego | Ego4D | 449 | 300 | 5.0 s (3.0-15.0) | 30 | 720x540 |
| synthetic | Cosmos | 445 | 400 | 5.0 s (all) | 24 | 1280x704 |

No clip is portrait, so the Point4D 518x294 caveat from Dyn-Bench does not arise here. The questions carry no
timestamps. 75 name a phase ("at the end", "at first", "after landing"); the rest are about the whole clip.

## 2. How it is scored

- The paper grades **free-text** answers with an LLM judge (o3 and o4-mini, disagreements resolved by hand).
  Each model is run with chain-of-thought (CoT) and direct-output prompts. AgentV5 answers with a letter, so a
  letter match makes the same decision without a judge (`d4rt_agent/agent_v5/vlm4d_report.py`).
- The columns of Table 1 are: real ego-centric, real exo-centric, real average, synthetic directional,
  synthetic FP, synthetic average, and overall, **weighted by question over all 1816**. Exo vs ego is split
  by id (real id <= 922 is exo), hard-coded in the repo's `acc_final_statistics.py`. Synthetic FP means the
  gold answer starts with "no".
- Reference points from Table 1 (CoT, overall): human 98.8, random 24.1, Gemini-2.5-Pro 62.0, GPT-4o 57.5,
  Qwen2.5-VL-72B 53.0, Qwen2.5-VL-7B 43.8. Re-scoring the repo's graded files with
  `taxonomy_report.paper_correct` reproduces each of these to 0.1.
- AgentV5's loader (`run.vlm4d_entries`) maps the gold text to a letter. Question ids are
  `vlm4d_<split>_<video>_<n>`, and the manifest here uses the same ids, so any AgentV5 run joins to these
  labels on `qid`.

## 3. The paper's taxonomy (Axis A)

The paper (Sec. 3.2, Fig. 2, supplementary Fig. A) sorts questions by **video source** (ego / exo /
synthetic) and by **annotation type**:

| type | what it asks | example |
|---|---|---|
| translational (TM) | which way something moves | "What direction is the horse in the foreground moving towards?" |
| rotational (RM) | which way something turns or spins, including "perspective-aware" turns in the subject's own frame | "Is the cat turning to the left or right from its own perspective?" |
| counting (STM) | how many objects do a motion, or how many times a motion happens | "How many times does the person look to the right before turning?" |
| false positive (FP) | whether the object or event exists at all | "Which direction did the dog move?" (no dog) |
| action (Fig. A only) | what someone does, or with which hand | "What is the left hand doing?" |

**The QA files do not ship these labels.** `taxonomy.paper_type` recovers them by rule. The rule's result
agrees with all 9 rows the dataset's preview parquet labels itself (`reasoning_focus`). On real video it
gives translational 54.3% / rotational 19.5% / counting 18.7%, against the paper's 55 / 19 / 17.

**False positives do not reproduce.** Only 56 real questions (4.1%) have a gold answer that denies the
object (47) or the event (9: "Nothing.", "Never picks up anything…", "did not use…"). The paper says 9%. On
synthetic video the count matches the repo's own `is_fp` exactly (45). Two things are probably mixed up:
**350 real questions *offer* an absence option**, but few have it as the answer. Either the paper counted
differently or questions changed after Fig. 2 was drawn. The released files cannot settle it.

## 4. The question shapes, by view (Axis B in plain words)

Counts are from `TAXONOMY_REPORT.md` §3. Kind names are AgentV5's (`d4rt_agent/agent_v5/kinds.py`);
**new** marks a kind AgentV5 does not have.

**Exo-centric (DAVIS, YouTube-VOS; 922 questions).** One subject, often in a group or a crowd.
- *Direction of travel* (`subject_direction`, 424): "What direction is the dog moving towards?" The options
  mix left / right / toward-or-away / not moving / absent, so one question tests several axes at once. 86 say
  "from the camera perspective".
- *Own-frame motion* (`traj_subject_own_frame`, 88): forwards or backwards, or turning left or right, "from
  its own perspective".
- *Turning seen from the camera, or with no frame named* (`subject_turn` **new**, 36): "Which way is the
  tractor turning towards?" The answer set adds "moving straight and not turning".
- *Spinning and circulating* (`rotation_sense` **new**, 86): "Is the dancer spinning clockwise or
  counter-clockwise?" and "Are the horses moving clockwise or counter-clockwise around the track?"
- *Range to the camera* (`cam_subject_range`, 72): "Is the dog coming closer or moving further away?"
- *Counting with a filter* (`count_with_predicate` **new**, 112): "How many people are moving to the left?"
  and "How many bears are facing away from the camera?"
- *Counting repetitions* (`repetition_count` **new**, 84): "How many times does the horse jump to the left?"
  and "How many full revolutions does the dancer spin clockwise?"

**Ego-centric (Ego4D; 449 questions).** Kitchens, workshops and sports seen from a head-mounted camera. This
is the most varied third of the benchmark: 422 distinct strings for 449 questions.
- *The wearer turns or walks* (`camera_path` 85, `camera_translation` 19): "Which way does the person turn
  to get the pan?" and "Which rotational direction does the person turn to go to the fridge?" (answer:
  clockwise). **"The person" is the camera wearer, so the subject is the camera.** 113 questions are flagged
  `ego_wearer`.
- *Objects moved by hand* (`subject_direction` 125, `landmark_relation` 25): "Which direction does he move the
  spoon to empty it into the bowl?" and "With respect to the sink, which direction does the person move the
  vegetable?"
- *Where things are or end up* (`relative_position` **new**, 55): "Where is the butter pack placed with
  respect to the bread slices?" and "On which side of the van does the cyclist go past?"
- *Hands* (`hand_attribution` **new**, 26): "Which hand does he use to drink from the cup?"
- *Repetitions, often filtered by hand or side* (`repetition_count` **new**, 61): "How many spoons of salt
  did he put in with his left hand?"
- *Semantic yes/no* (`semantic`, 9) and *object identity by motion* (`other`, 9): "Does the basketball hit
  the backboard?" and "What object is moving to the left in the scene?"

**Synthetic (Cosmos; 445 questions).** 420 follow one template. 436 are `subject_direction`. 380 of the 445
offer left / right / not moving / "no X there", and 35 offer toward / away instead of left / right. The few
outliers ask about rotation (9, `rotation_sense`).

## 5. Conventions and pitfalls

1. **Ego "person" = camera.** A turn of the person in an Ego4D clip is the camera's yaw. A walk is the
   camera's translation. "Clockwise" for a turn presumably means seen from above, i.e. turning right. That
   reading is an assumption until it is checked against measured camera yaw. AgentV5's family menu describes
   `camera` as "only the camera itself", and Qwen is never told the clip is ego-centric.
2. **Most direction questions name no frame.** 884 of the 985 `subject_direction` questions say neither
   "camera" nor "own". In the paper's runs, the 86 real questions that *do* say "from the camera perspective"
   are harder: Qwen2.5-VL-7B gets 33.7% of them, against 45.5% of the frame-free ones. Dyn-Bench's
   calibration found that left/right followed the first camera's axes, which here is the same as the image
   for a static camera. Whether VLM4D annotators meant the image or the world is untested.
3. **Absence options are offered often and rarely right.** 795 questions offer one (all 445 synthetic, 350
   real), and it is the answer in 92 (12%). 21 absence options name the wrong object ("there are no buses in
   the video" on a kayak question, "no koala there" on a tree question) or misspell it. An existence check has
   to read the option's noun, and has to be calibrated against a 12% base rate: an eager "it's not there"
   loses.
4. **Static answers are common.** "not moving" is offered in 911 questions and is the answer in 165 (18%).
   Static-type answers ("not moving", "not spinning", "moving straight and not turning", "staying in place")
   are the gold for 236 of the 653 real questions that offer one (36%).
5. **Counting answers are small, and often zero.** 77 of 256 counting questions have "0" as the answer.
   Direction-filtered counts ("how many … to the left") are frequently zero because everything moves the
   other way.
6. **Answer priors are strong.** A text-only baseline, which picks the option wording most often correct
   elsewhere in the benchmark (leave-one-out) and never sees the video, scores **47.2% overall**. That beats
   Qwen2.5-VL-7B (43.8) and the mean of the 23 graded models (44.2). On synthetic video it scores **61.1%**,
   because "left" is the answer to 198 of 445 questions. Any VLM4D result has to be read against this
   baseline, not against 25% random.
7. **Typos everywhere**: "bicyclec", "mototrbikes", "clckswise", "plances", "helicoptor", "rokect". AgentV5's
   slot check (D3), which rejects names that are not in the question, has to accept the misspelt noun.
8. One question has two options with the same text; either letter counts (`answer_also`).

## 6. Compared with the other two benchmarks

| | DSI-Bench | Dyn-Bench | VLM4D |
|---|---|---|---|
| questions | 1769 | 7000 | 1816 |
| time in questions | none | timestamps on 16 generator frames | none; 1-15 s clips, one action |
| dominant question | own-frame trajectory (AgentV4's bucket) | camera motion, distance, scene relations | direction of travel in an unnamed frame |
| measurement in metres | no | yes (speed, distance) | none |
| distinctive | 4 confirmed buckets | 9 paper sub-tasks | ego-wearer questions, false positives, counting repetitions |
| AgentV5 kinds missing | 0 | 0 (built for it) | 6 kinds, 490 questions (27%) |

## 7. Reproduce

```bash
python -m vlm4d_dataset_exploration.vlm4d_data manifest            # labels only, seconds
sbatch --time=00:20:00 --cpus-per-task=1 --mem-per-cpu=2G scripts/run_cpu_task.slurm \
       -m vlm4d_dataset_exploration.vlm4d_data videos               # videos.json (1000 mp4 headers)
python -m vlm4d_dataset_exploration.taxonomy_report                # manifest.json + TAXONOMY_REPORT.md
python -m pytest vlm4d_dataset_exploration                         # 49 GPU-free tests
```

The report uses the `spatialstack-qwen35` env for pandas (the preview check); everything else is stdlib.

**How good the rules are.** I hand-audited a stratified sample of 131 questions (up to 8 per kind, seed
20261013) and agreed with 124 labels (95%). Two of the seven misses were one rule (a synthetic "turn" offered
with clockwise options), since fixed. Five remain as known ambiguities: "which side does he cut first"
(`event_order`, arguably `relative_position`), a phone moving closer to the ego wearer (`pair_relation`,
arguably `cam_subject_range`), page flips filed as `rotation_sense`, and "did the shuttlecock go higher than
the camera" (`subject_direction`, arguably AgentV5's `cam_subject_elevation`).
