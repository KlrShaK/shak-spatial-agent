# DSI-Bench grounding-tool comparison

## Run identity

- Implementation commit: `56d455caa1a7868aa99aec237cbd541b40157f7d`
- Slurm job(s): `['9086647']` on `['eu-a65-07']`
- GPU: `['NVIDIA A100 80GB PCIe']`
- Fixed manifest: 25 questions from `DSI-Bench` `std`

## Headline results

- Current: 9/25
- Previous agent: 10/25
- Failed forced-grounding run: 7/25 with 4 unanswered/failed
- Unchanged Qwen baseline: 13/25
- Completion: 22 strict, 0 relaxed, 3 failed, 0 infrastructure errors
- Completion rates: 88.0% strict, 0.0% relaxed, 12.0% failed
- Accuracy among completed statuses: 9/22 (40.9%)

## Tool and orchestration metrics

- Grounding: 60 calls, 60 inferences, 0 cache hits, 19 not found, 0 malformed
- Grounding modes: {'bbox': 29, 'points': 31}
- Downstream calls: 66 D4RT, 12 math, 65 rejected
- Steps: 225 total, 9.00 mean, 28 max
- Discarded-suffix turns: 0/225 (0.0%)
- D4RT visibility by grounding mode: {'bbox': {'mean': 0.677734375, 'count': 48}, 'points': {'mean': 0.625, 'count': 18}}
- Runtime: 53.5 minutes total, 128.3 seconds/question

## Per task

| Task | Correct | Total | Accuracy | Statuses |
| --- | ---: | ---: | ---: | --- |
| Cam:dynamic scene | 1 | 4 | 25.0% | {'complete': 3, 'failed': 1} |
| Cam:static scene | 0 | 4 | 0.0% | {'failed': 1, 'complete': 3} |
| Obj-Cam distance | 3 | 5 | 60.0% | {'complete': 4, 'failed': 1} |
| Obj-Cam orientation | 2 | 4 | 50.0% | {'complete': 4} |
| Obj:moving cam | 1 | 4 | 25.0% | {'complete': 4} |
| Obj:static cam | 2 | 4 | 50.0% | {'complete': 4} |

## Answer flips

- Versus previous_agent: 4 gains, 5 losses, 4 changed-wrong answers.
  - `CameraBench_c0_camerabench_m0jmssq5ptw.3.12_ccd85243`: A → B (GT B, gain)
  - `CameraBench_c1_656086265ccda299e2c2ec394f4bfded4daa3ea64efcb4ee6c95b8c7de.0_95469228`: C → D (GT C, loss)
  - `CameraBench_c2_camerabench_u35wis62r2m.1.4_8c093924`: A → — (GT C, changed_wrong)
  - `CameraBench_c3_camerabench_3m-me9axcto.1.1_60268455`: D → C (GT D, loss)
  - `SynFMC_c3_synfmc_rendered_traj_results_dynamic_45_video_480p_0fd58550`: D → — (GT A, changed_wrong)
  - `SynFMC_c5_synfmc_rendered_traj_results_dynamic_14_video_480p_72d9da62`: C → A (GT C, loss)
  - `internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874`: C → A (GT C, loss)
  - `internet_c4_internet_an2sz4h4qzk_video-scene-00048_265c8450`: A → D (GT D, gain)
  - `k700_c0_k700_paragliding_95df0yxigbw_000076_000086_video-scene-00000_a1d4095b`: C → A (GT A, gain)
  - `k700_c1_700_playing_polo_j7j8_hz_hx8_000018_000028_video-scene-00001_68f048f4`: A → D (GT D, gain)
  - `llava178k_c4_videos_youtube_video_2024_ytb_pcj0xjv7gi0_video-scene-00002_b63a5bf6`: C → — (GT D, changed_wrong)
  - `llava178k_c1_videos_youtube_video_2024_ytb_pzg1hdiwfyk_video-scene-00002_fb3c4084`: A → D (GT A, loss)
  - `llava178k_c2_videos_youtube_video_2024_ytb_opt9l_7g68m_video-scene-00004_11dfd093`: C → D (GT B, changed_wrong)
- Versus failed_forced_grounding: 3 gains, 1 losses, 10 changed-wrong answers.
  - `CameraBench_c1_656086265ccda299e2c2ec394f4bfded4daa3ea64efcb4ee6c95b8c7de.0_95469228`: — → D (GT C, changed_wrong)
  - `CameraBench_c2_camerabench_u35wis62r2m.1.4_8c093924`: B → — (GT C, changed_wrong)
  - `SynFMC_c3_synfmc_rendered_traj_results_dynamic_45_video_480p_0fd58550`: D → — (GT A, changed_wrong)
  - `SynFMC_c5_synfmc_rendered_traj_results_dynamic_14_video_480p_72d9da62`: B → A (GT C, changed_wrong)
  - `internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874`: B → A (GT C, changed_wrong)
  - `internet_c5_internet_0knwuip85a8_video-scene-00009_39b26678`: — → D (GT D, gain)
  - `k700_c4_k700_jogging_qy8rjbxblna_000116_000126_video-scene-00001_ab85ad6e`: — → C (GT A, changed_wrong)
  - `k700_c5_700_snowboarding_9o2tvonlrmu_000003_000013_video-scene-00000_227e207a`: B → D (GT C, changed_wrong)
  - `k700_c1_700_playing_polo_j7j8_hz_hx8_000018_000028_video-scene-00001_68f048f4`: A → D (GT D, gain)
  - `llava178k_c4_videos_youtube_video_2024_ytb_pcj0xjv7gi0_video-scene-00002_b63a5bf6`: A → — (GT D, changed_wrong)
  - `llava178k_c5_videos_youtube_video_2024_ytb_16mzjk4aojs_video-scene-00000_b36d2eb1`: — → B (GT B, gain)
  - `llava178k_c0_videos_youtube_video_2024_ytb_djaut-hx2iu_video-scene-00000_bf9d0572`: B → C (GT B, loss)
  - `llava178k_c1_videos_youtube_video_2024_ytb_pzg1hdiwfyk_video-scene-00002_fb3c4084`: B → D (GT A, changed_wrong)
  - `llava178k_c2_videos_youtube_video_2024_ytb_opt9l_7g68m_video-scene-00004_11dfd093`: C → D (GT B, changed_wrong)
- Versus baseline: 3 gains, 7 losses, 7 changed-wrong answers.
  - `CameraBench_c1_656086265ccda299e2c2ec394f4bfded4daa3ea64efcb4ee6c95b8c7de.0_95469228`: C → D (GT C, loss)
  - `CameraBench_c2_camerabench_u35wis62r2m.1.4_8c093924`: B → — (GT C, changed_wrong)
  - `CameraBench_c3_camerabench_3m-me9axcto.1.1_60268455`: D → C (GT D, loss)
  - `SynFMC_c3_synfmc_rendered_traj_results_dynamic_45_video_480p_0fd58550`: D → — (GT A, changed_wrong)
  - `SynFMC_c5_synfmc_rendered_traj_results_dynamic_14_video_480p_72d9da62`: C → A (GT C, loss)
  - `internet_c2_internet_2d0iwohbsju_video-scene-00020_fdbaa511`: C → D (GT B, changed_wrong)
  - `internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874`: C → A (GT C, loss)
  - `internet_c4_internet_an2sz4h4qzk_video-scene-00048_265c8450`: C → D (GT D, gain)
  - `internet_c5_internet_0knwuip85a8_video-scene-00009_39b26678`: A → D (GT D, gain)
  - `internet_c0_internet_fopbvllxkz0_video-scene-00033_c8413d28`: A → C (GT B, changed_wrong)
  - `k700_c4_k700_jogging_qy8rjbxblna_000116_000126_video-scene-00001_ab85ad6e`: D → C (GT A, changed_wrong)
  - `k700_c5_700_snowboarding_9o2tvonlrmu_000003_000013_video-scene-00000_227e207a`: B → D (GT C, changed_wrong)
  - `k700_c0_k700_paragliding_95df0yxigbw_000076_000086_video-scene-00000_a1d4095b`: C → A (GT A, gain)
  - `llava178k_c4_videos_youtube_video_2024_ytb_pcj0xjv7gi0_video-scene-00002_b63a5bf6`: D → — (GT D, loss)
  - `llava178k_c0_videos_youtube_video_2024_ytb_djaut-hx2iu_video-scene-00000_bf9d0572`: B → C (GT B, loss)
  - `llava178k_c1_videos_youtube_video_2024_ytb_pzg1hdiwfyk_video-scene-00002_fb3c4084`: B → D (GT A, changed_wrong)
  - `llava178k_c2_videos_youtube_video_2024_ytb_opt9l_7g68m_video-scene-00004_11dfd093`: B → D (GT B, loss)

## Per question

| Question | Task | Current | Previous | Failed prompt | Baseline | GT | Status | Steps | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| CameraBench_c0_camerabench_m0jmssq5ptw.3.12_ccd85243 | Obj:static cam | B | A | B | B | B | complete | 6 | 112.48 |
| CameraBench_c1_656086265ccda299e2c2ec394f4bfded4daa3ea64efcb4ee6c95b8c7de.0_95469228 | Obj:moving cam | D | C | — | C | C | complete | 6 | 49.29 |
| CameraBench_c2_camerabench_u35wis62r2m.1.4_8c093924 | Cam:static scene | — | A | B | B | C | failed | 28 | 745.9 |
| CameraBench_c3_camerabench_3m-me9axcto.1.1_60268455 | Cam:dynamic scene | C | D | C | D | D | complete | 5 | 64.4 |
| CameraBench_c4_camerabench_-2uia-xmjc0.5.3_877d2d07 | Obj-Cam distance | A | A | A | A | A | complete | 5 | 35.5 |
| SynFMC_c1_synfmc_rendered_traj_results_dynamic_6_video_480p_728b5ef1 | Obj:moving cam | C | C | C | C | D | complete | 6 | 65.51 |
| SynFMC_c2_synfmc_rendered_traj_results_dynamic_450_video_480p_868f34a5 | Cam:static scene | B | B | B | B | A | complete | 6 | 50.41 |
| SynFMC_c3_synfmc_rendered_traj_results_dynamic_45_video_480p_0fd58550 | Cam:dynamic scene | — | D | D | D | A | failed | 28 | 257.84 |
| SynFMC_c4_synfmc_rendered_traj_results_dynamic_183_video_480p_e9b12cbc | Obj-Cam distance | B | B | B | B | B | complete | 4 | 34.62 |
| SynFMC_c5_synfmc_rendered_traj_results_dynamic_14_video_480p_72d9da62 | Obj-Cam orientation | A | C | B | C | C | complete | 6 | 61.24 |
| internet_c2_internet_2d0iwohbsju_video-scene-00020_fdbaa511 | Cam:static scene | D | D | D | C | B | complete | 12 | 303.41 |
| internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874 | Cam:dynamic scene | A | C | B | C | C | complete | 7 | 81.63 |
| internet_c4_internet_an2sz4h4qzk_video-scene-00048_265c8450 | Obj-Cam distance | D | A | D | C | D | complete | 5 | 35.2 |
| internet_c5_internet_0knwuip85a8_video-scene-00009_39b26678 | Obj-Cam orientation | D | D | — | A | D | complete | 12 | 115.32 |
| internet_c0_internet_fopbvllxkz0_video-scene-00033_c8413d28 | Obj:static cam | C | C | C | A | B | complete | 3 | 31.11 |
| k700_c3_k700_skiing_mono_rqfndqdk5ny_000080_000090_video-scene-00000_fd807178 | Cam:dynamic scene | B | B | B | B | B | complete | 12 | 196.06 |
| k700_c4_k700_jogging_qy8rjbxblna_000116_000126_video-scene-00001_ab85ad6e | Obj-Cam distance | C | C | — | D | A | complete | 3 | 37.52 |
| k700_c5_700_snowboarding_9o2tvonlrmu_000003_000013_video-scene-00000_227e207a | Obj-Cam orientation | D | D | B | B | C | complete | 14 | 343.89 |
| k700_c0_k700_paragliding_95df0yxigbw_000076_000086_video-scene-00000_a1d4095b | Obj:static cam | A | C | A | C | A | complete | 6 | 100.38 |
| k700_c1_700_playing_polo_j7j8_hz_hx8_000018_000028_video-scene-00001_68f048f4 | Obj:moving cam | D | A | A | D | D | complete | 6 | 77.36 |
| llava178k_c4_videos_youtube_video_2024_ytb_pcj0xjv7gi0_video-scene-00002_b63a5bf6 | Obj-Cam distance | — | C | A | D | D | failed | 28 | 225.85 |
| llava178k_c5_videos_youtube_video_2024_ytb_16mzjk4aojs_video-scene-00000_b36d2eb1 | Obj-Cam orientation | B | B | — | B | B | complete | 6 | 68.44 |
| llava178k_c0_videos_youtube_video_2024_ytb_djaut-hx2iu_video-scene-00000_bf9d0572 | Obj:static cam | C | C | B | B | B | complete | 3 | 36.12 |
| llava178k_c1_videos_youtube_video_2024_ytb_pzg1hdiwfyk_video-scene-00002_fb3c4084 | Obj:moving cam | D | A | B | B | A | complete | 3 | 30.22 |
| llava178k_c2_videos_youtube_video_2024_ytb_opt9l_7g68m_video-scene-00004_11dfd093 | Cam:static scene | D | C | C | B | B | complete | 5 | 47.94 |

## Visual review and interpretation

All 25 per-question grounding sheets and all 41 exact-frame overlays covering
the requested question/source-frame pairs were inspected. The report also
records all 19 `not_found` results without drawing invented geometry.

Whole-object grounding was usually plausible. Representative examples include
the [runner](groundings/camerabench_m0jmssq5ptw.3.12_ccd85243.jpg), the
[police car across six source frames](groundings/internet_0knwuip85a8_video-scene-00009_39b26678.jpg),
the [closest white horse](groundings/internet_an2sz4h4qzk_video-scene-00048_265c8450.jpg),
and the [harvester](groundings/videos_youtube_video_2024_ytb_16mzjk4aojs_video-scene-00000_b36d2eb1.jpg).
The main severe object-box failure is the
[closest red car](groundings/videos_youtube_video_2024_ytb_pcj0xjv7gi0_video-scene-00002_b63a5bf6.jpg):
the returned box covers the ego vehicle's hood/dashboard rather than the red
car ahead.

Point grounding remained much less reliable than ordinary boxes. In several
small or one-sided views, nominal chest/back points are adjacent points on the
same visible surface rather than evidence of two anatomically distinguishable
parts. This affects the
[yellow-jacket person](groundings/656086265ccda299e2c2ec394f4bfded4daa3ea64efcb4ee6c95b8c7de.0_95469228.jpg),
the [pink character](groundings/synfmc_rendered_traj_results_dynamic_6_video_480p_728b5ef1.jpg),
and the [snowboarder](groundings/700_snowboarding_9o2tvonlrmu_000003_000013_video-scene-00000_227e207a.jpg).
Background requests also often returned `not_found`; when points were returned,
they could be tightly clustered instead of spanning the scene, as in the
[Minecraft building request](groundings/videos_youtube_video_2024_ytb_opt9l_7g68m_video-scene-00004_11dfd093.jpg).

## Failure taxonomy

The dominant failure assigned to each of the 16 wrong or unanswered examples
is below. These labels identify the first decisive trace failure; later failures
can compound it.

| Question | Dominant failure | Trace/overlay evidence |
| --- | --- | --- |
| `CameraBench_c1_656086...` | part/point localization | The correctly boxed tiny person received chest/back points only 8 normalized y-units apart; the resulting facing axis selected backward instead of forward. |
| `CameraBench_c2_camerabench_u35...` | orchestration/termination | End-frame visibility was zero, after which strict and relaxed attempts repeatedly emitted invalid, unfinished action text and never finalized. |
| `CameraBench_c3_camerabench_3m...` | D4RT visibility/tracking | The only returned star points came from frame 9 and all three were invisible at queried frames 0 and 31; the model then made an unsupported visual guess. |
| `SynFMC_c1_...dynamic_6...` | part/point localization | The whole-character box is sound, but adjacent one-view chest/back points define the wrong facing direction and flip forward-left to backward-right. |
| `SynFMC_c2_...dynamic_450...` | wrong semantic request | Three rigid-background requests returned `not_found`; the agent substituted moving Sonic as a camera-motion proxy and selected backward. |
| `SynFMC_c3_...dynamic_45...` | orchestration/termination | Static landmarks were not found, the foreground character was invisible at the end, and the agent repeated the same start-frame query until no final action remained. |
| `SynFMC_c5_...dynamic_14...` | part/point localization | A one-view chest/back pair produced an unreliable facing axis and the wrong left/right-to-back mapping. |
| `internet_c2_...2d0iwohbsju...` | D4RT visibility/tracking | Three ceiling points were valid only at frame 0 and all invisible at frame 31; eight malformed/repeated calculations followed before a visual guess. |
| `internet_c3_...4cz3oqlfupm...` | coordinate-frame reasoning | Only the deer at source frame 2 was visible; the agent treated its negative absolute y-coordinate as evidence that the camera moved downward. |
| `internet_c0_...fopbvllxkz0...` | D4RT visibility/tracking | The car box is plausible, but its end-frame track is invisible; the agent converted missing evidence into “basically unchanged.” |
| `k700_c4_...jogging...` | D4RT visibility/tracking | The runner is boxed correctly but invisible at frame 31, so the agent returns “cannot be determined” instead of the benchmark's unchanged option. |
| `k700_c5_...snowboarding...` | trajectory calculation | One part track is invisible at the end; ten calculator calls repeat a forbidden conditional before the agent submits an unsupported orientation answer. |
| `llava178k_c4_...pcj0xjv7gi0...` | bbox localization | The box selects the ego hood/dashboard, not the red car; subsequent queries are invisible/rejected and neither attempt finalizes. |
| `llava178k_c0_...djaut-hx2iu...` | option selection | The pickup is boxed correctly but invisible at the end; leaving the field of view is incorrectly equated with moving backward. |
| `llava178k_c1_...pzg1hdiwfyk...` | D4RT visibility/tracking | The man is boxed correctly, but the tracked vertical change is nearly zero and the agent selects unchanged rather than downward. |
| `llava178k_c2_...opt9l_7g68m...` | ambiguous/static-background choice | Three supposed building corners are clustered near one image region, including two nearly identical points, making the inferred egomotion and option unreliable. |

Across these examples, the dominant remaining bottlenecks are therefore not
ordinary whole-object detection alone. They are partial visibility, D4RT track
coverage, unobservable one-frame part requests, camera/facing-axis reasoning,
and failure to adapt after a rejected or uninformative result.

## Orchestration audit

The CPU audit passed all 25 records and 225 model turns with zero structural
violations:

- every effective turn contains exactly one action;
- no discarded suffix was placed in effective history (`0/225`);
- every accepted action has valid immutable ledger provenance;
- rejected actions create no evidence;
- every D4RT query refers to an existing same-clip `qg_N`;
- bbox queries use the host `ensemble5` policy and point queries preserve
  explicit point IDs.

This confirms that the host now stops and executes after one action. It does not
mean the model always adapts: 65 actions were rejected, and repeated malformed
or semantically identical retries caused all three unanswered cases.

## Hypothesis assessment

The hypothesis is **partially supported at the grounding boundary but weakened
as an end-to-end performance hypothesis**.

The isolated context produced strictly parseable grounding outputs (60/60, no
malformed grounder responses), and visual review shows that most ordinary
whole-object boxes are reasonable. The single-action/ledger architecture is
also structurally clean. However, end-to-end accuracy fell from 10/25 to 9/25,
completion fell from 25/25 to 22/25, and the result remains below the unchanged
Qwen baseline at 13/25. Four gains over the previous agent were offset by five
losses and four changed-wrong answers.

Isolation therefore addresses context-contaminated coordinate generation, but
does not by itself solve the benchmark. The strongest next generic improvements
would be better rejection recovery and termination behavior, visibility-aware
source/target selection, and a representation for orientation that does not ask
a single 2D view to identify simultaneously hidden chest/back surfaces.

## Reproducibility and limitations

- The implementation evaluated by every answer record is commit
  `56d455caa1a7868aa99aec237cbd541b40157f7d`, seed 42, max 14 actions per
  strict/relaxed attempt, and D4RT `ensemble5`.
- The user later overrode the plan's Blackwell-only rule with an A100-80GB versus
  RTX PRO 6000 race. Job `9086647` won the race and ran all 25 examples on an
  NVIDIA A100 80GB PCIe; Blackwell candidate `9086639` was cancelled before
  producing evaluation outputs.
- The manifest SHA-256 is
  `4edae52a6a932579c5dc7dbf625e70ba26f9589334cd878ea517c8cfb4e7359c`,
  identical to the previous and baseline runs.
- There are no DSI-Bench ground-truth boxes or point annotations. Overlay
  judgments are qualitative and must not be interpreted as IoU accuracy.
- This is one deterministic-seed run on 25 fixed questions, so question-level
  flips are more informative than small aggregate differences.
- Zero cache hits means no exact canonical grounding request repeated in this
  run; cache behavior is covered by CPU tests, not exercised by these 25 traces.
- The evaluation intentionally retains generic tools and does not add
  task-specific recovery, answer rules, forced finalization, or DSI-specific
  heuristics.
