# Plain-Qwen grounding probe — findings

## Experiment

The 25 saved DSI-Bench agent traces contain 138 requested `query_d4rt`
groundings: 135 executed requests and 3 requests rejected by the host. Repeated
requests collapse to 56 unique `(video, sampled source frame, trace label)`
instances.

For each unique instance, Qwen3-VL-8B-Instruct received:

- exactly one sampled source-frame image;
- the label copied verbatim from the agent trace; and
- one direct instruction to return a tight
  `[x_min,y_min,x_max,y_max]` box normalized to `[0,1000]`, or `null` if absent.

It received no DSI question, other video frames, tool schema, D4RT output, or
conversation history. Decoding was greedy with seed 42. D4RT was not loaded.
The GPU pass was SLURM job `8878085`.

## Result

All 56 probes completed with strict, parseable JSON:

| Outcome | Count |
| --- | ---: |
| Bounding box | 50 |
| Explicit `null` | 6 |
| Parse/runtime error | 0 |

The standalone boxes are **not reproductions of the agent-loop boxes**:

| Comparison with trace box | Result |
| --- | ---: |
| Median IoU, 56 unique probes | **0.013** |
| Mean IoU, detected probes | **0.129** |
| IoU ≥ 0.50 | **3 / 50** |
| IoU ≥ 0.75 | **0 / 50** |
| Median center displacement in normalized coordinates | **152.1** |

Counting repeated trace occurrences instead of unique probes gives median IoU
0.017 and only 4/128 detected occurrences at IoU ≥ 0.50. IoU here measures
agreement between two Qwen prompting contexts, not accuracy: DSI-Bench provides
no ground-truth boxes.

## Visual diagnosis

The overlays support a mixed but actionable conclusion.

### Obvious whole objects often improve under the isolated prompt

The standalone prompt tightly localizes the distant runner and the person on
the ice. Their standalone boxes have zero overlap with the rough boxes generated
inside the agent loop. In both cases the isolated box is visually on the target,
whereas the trace box is displaced. The same is true for several large,
unambiguous subjects such as the CameraBench character.

Examples:

- [distant runner](frames/CameraBench_c0_camerabench_m0jmssq5ptw.3.12_ccd85243_f00.jpg)
- [person on ice](frames/CameraBench_c1_656086265ccda299e2c2ec394f4bfded4daa3ea64efcb4ee6c95b8c7de.0_95469228_f00.jpg)
- [CameraBench character](frames/CameraBench_c4_camerabench_-2uia-xmjc0.5.3_877d2d07_f00.jpg)

This means the exact grounding failures from the full reasoning loop do **not**
appear automatically when the same model is used as a direct detector. The
long video/reasoning context and hand-written coarse coordinates are part of
the problem.

### Fine-grained front/rear grounding remains unreliable

The body-part probes usually return strongly overlapping chest/back boxes,
often on essentially the same upper-body region. Across runner, synthetic
character, man, paraglider, horse, and skier, the median chest/back box IoU is
**0.568**. A front/rear axis built from the centroids of such boxes is poorly
conditioned even when both detections look superficially plausible.

Examples:

- [synthetic character](frames/SynFMC_c1_synfmc_rendered_traj_results_dynamic_6_video_480p_728b5ef1_f00.jpg)
- [paraglider](frames/k700_c0_k700_paragliding_95df0yxigbw_000076_000086_video-scene-00000_a1d4095b_f00.jpg)
- [polo horse](frames/k700_c1_700_playing_polo_j7j8_hz_hx8_000018_000028_video-scene-00001_68f048f4_f00.jpg)
- [skier](frames/k700_c5_700_snowboarding_9o2tvonlrmu_000003_000013_video-scene-00000_227e207a_f00.jpg)

The harvester is the exception: its front and back prompts select visibly
different machine regions, although the meaning of those regions is still
somewhat prompt-dependent.

### Ambiguous background labels still fail

All six `null` replies are scene landmarks: five tree descriptions from one
blurred clip and one background pole. The tree frame visibly contains foliage,
but labels such as “a different tree,” “tree in the center,” and “tree on the
right” are not reliably referential from the single image. The one returned
“rock” box selects an ambiguous dark region.

Examples:

- [ambiguous trees and rock](frames/internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874_f00.jpg)
- [ceiling decorations](frames/CameraBench_c3_camerabench_3m-me9axcto.1.1_60268455_f00.jpg)

Relative labels such as near/far ceiling decorations also collapse onto nearly
the same region. Static-background camera-motion recipes therefore remain
vulnerable even with the direct detection prompt.

## Conclusion

This probe does **not** support the strongest hypothesis that Qwen inevitably
produces the same erroneous boxes on these images. The isolated detector often
chooses a very different—and for obvious whole objects visibly better—box.

It **does** confirm that grounding is a major failure source for the operations
the agent most depends on:

1. fine-grained front-versus-rear body points are not reliably separated;
2. relative or generic background landmarks are poorly specified; and
3. grounding quality is highly sensitive to prompting context.

The experiment cannot by itself attribute D4RT's 54% target invisibility to
grounding, because it did not rerun D4RT with the new boxes. The clean next
causal test is to replay the original D4RT queries with these standalone boxes
and compare source visibility, target visibility, spread, and final answer
changes against the saved run.

See [SUMMARY.md](SUMMARY.md) for all 56 boxes and frame links. Raw replies are
under `detections/`, combined new-box overlays are under `frames/`, and
`requests.json` maps every unique probe back to all 138 trace steps.
