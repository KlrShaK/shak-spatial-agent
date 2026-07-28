# Plain-Qwen grounding debug

This probe isolates Qwen's 2D grounding from the D4RT reasoning loop. For each
unique `(question, t_src, label)` requested in the saved agent traces, Qwen saw
only that one sampled image and the same fixed object-detection prompt. It received
no video question, other frames, tool schema, D4RT output, or conversation history.

Repeated trace requests with identical image and label were evaluated once and
mapped back to every occurrence. Each saved source-frame image combines all new
plain-Qwen detections requested on that frame. It does not draw the original
agent boxes.

## Status

| Metric | Value |
| --- | ---: |
| Questions | 25 |
| Original grounding requests | 138 |
| Unique image/label probes | 56 |
| Completed probes | 56 |
| Parsed boxes | 50 |
| Explicit not-visible responses | 6 |
| Errors | 0 |
| Median IoU with original agent box | 0.013 |
| IoU ≥ 0.50 | 3 / 50 |
| IoU ≥ 0.75 | 0 / 50 |
| Median box-center distance (0–1000 coordinates) | 152.1 |

> IoU measures whether two Qwen prompting contexts chose the same region; it is
> not localization accuracy because DSI-Bench supplies no ground-truth boxes.
> The overlays must be inspected to decide whether agreement reproduces a mistake.

## Per question

| Question | Dataset | Trace requests | Unique probes | Detected | Median IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| `CameraBench_c0_camerabench_m0jmssq5ptw.3.12_ccd85243` | CameraBench | 3 | 3 | 3 | 0.000 |
| `CameraBench_c1_656086265ccda299e2c2ec394f4bfded4daa3ea64efcb4ee6c95b8c7de.0_95469228` | CameraBench | 13 | 1 | 1 | 0.000 |
| `CameraBench_c2_camerabench_u35wis62r2m.1.4_8c093924` | CameraBench | 11 | 1 | 1 | 0.000 |
| `CameraBench_c3_camerabench_3m-me9axcto.1.1_60268455` | CameraBench | 8 | 4 | 4 | 0.000 |
| `CameraBench_c4_camerabench_-2uia-xmjc0.5.3_877d2d07` | CameraBench | 11 | 1 | 1 | 0.108 |
| `SynFMC_c1_synfmc_rendered_traj_results_dynamic_6_video_480p_728b5ef1` | SynFMC | 3 | 3 | 3 | 0.000 |
| `SynFMC_c2_synfmc_rendered_traj_results_dynamic_450_video_480p_868f34a5` | SynFMC | 11 | 2 | 2 | 0.142 |
| `SynFMC_c3_synfmc_rendered_traj_results_dynamic_45_video_480p_0fd58550` | SynFMC | 11 | 3 | 3 | 0.000 |
| `SynFMC_c4_synfmc_rendered_traj_results_dynamic_183_video_480p_e9b12cbc` | SynFMC | 2 | 1 | 1 | 0.191 |
| `SynFMC_c5_synfmc_rendered_traj_results_dynamic_14_video_480p_72d9da62` | SynFMC | 3 | 3 | 3 | 0.416 |
| `internet_c0_internet_fopbvllxkz0_video-scene-00033_c8413d28` | internet | 5 | 5 | 5 | 0.330 |
| `internet_c2_internet_2d0iwohbsju_video-scene-00020_fdbaa511` | internet | 2 | 1 | 1 | 0.044 |
| `internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874` | internet | 11 | 6 | 1 | 0.000 |
| `internet_c4_internet_an2sz4h4qzk_video-scene-00048_265c8450` | internet | 2 | 1 | 1 | 0.000 |
| `internet_c5_internet_0knwuip85a8_video-scene-00009_39b26678` | internet | 5 | 1 | 1 | 0.149 |
| `k700_c0_k700_paragliding_95df0yxigbw_000076_000086_video-scene-00000_a1d4095b` | k700 | 3 | 3 | 3 | 0.000 |
| `k700_c1_700_playing_polo_j7j8_hz_hx8_000018_000028_video-scene-00001_68f048f4` | k700 | 3 | 3 | 3 | 0.000 |
| `k700_c3_k700_skiing_mono_rqfndqdk5ny_000080_000090_video-scene-00000_fd807178` | k700 | 4 | 2 | 2 | 0.353 |
| `k700_c4_k700_jogging_qy8rjbxblna_000116_000126_video-scene-00001_ab85ad6e` | k700 | 2 | 1 | 1 | 0.084 |
| `k700_c5_700_snowboarding_9o2tvonlrmu_000003_000013_video-scene-00000_227e207a` | k700 | 3 | 3 | 3 | 0.000 |
| `llava178k_c0_videos_youtube_video_2024_ytb_djaut-hx2iu_video-scene-00000_bf9d0572` | llava178k | 2 | 2 | 1 | 0.712 |
| `llava178k_c1_videos_youtube_video_2024_ytb_pzg1hdiwfyk_video-scene-00002_fb3c4084` | llava178k | 2 | 1 | 1 | 0.208 |
| `llava178k_c2_videos_youtube_video_2024_ytb_opt9l_7g68m_video-scene-00004_11dfd093` | llava178k | 2 | 1 | 1 | 0.009 |
| `llava178k_c4_videos_youtube_video_2024_ytb_pcj0xjv7gi0_video-scene-00002_b63a5bf6` | llava178k | 13 | 1 | 1 | 0.158 |
| `llava178k_c5_videos_youtube_video_2024_ytb_16mzjk4aojs_video-scene-00000_b36d2eb1` | llava178k | 3 | 3 | 3 | 0.010 |

## Probe details

| Question / frame | Label | Repetitions | Qwen box | IoU | Frame detections |
| --- | --- | ---: | --- | ---: | --- |
| `CameraBench_c0_camerabench_m0jmssq5ptw.3.12_ccd85243` / 0 | runner | 1 | `[447.0, 350.0, 475.0, 440.0]` | 0.000 | [view](frames/CameraBench_c0_camerabench_m0jmssq5ptw.3.12_ccd85243_f00.jpg) |
| `CameraBench_c0_camerabench_m0jmssq5ptw.3.12_ccd85243` / 0 | runner's back | 1 | `[450.0, 350.0, 475.0, 437.0]` | 0.000 | [view](frames/CameraBench_c0_camerabench_m0jmssq5ptw.3.12_ccd85243_f00.jpg) |
| `CameraBench_c0_camerabench_m0jmssq5ptw.3.12_ccd85243` / 0 | runner's chest | 1 | `[458.0, 367.0, 475.0, 400.0]` | 0.000 | [view](frames/CameraBench_c0_camerabench_m0jmssq5ptw.3.12_ccd85243_f00.jpg) |
| `CameraBench_c1_656086265ccda299e2c2ec394f4bfded4daa3ea64efcb4ee6c95b8c7de.0_95469228` / 0 | person | 13 | `[503.0, 527.0, 527.0, 600.0]` | 0.000 | [view](frames/CameraBench_c1_656086265ccda299e2c2ec394f4bfded4daa3ea64efcb4ee6c95b8c7de.0_95469228_f00.jpg) |
| `CameraBench_c2_camerabench_u35wis62r2m.1.4_8c093924` / 0 | corner of the building structure | 11 | `[738.0, 333.0, 788.0, 400.0]` | 0.000 | [view](frames/CameraBench_c2_camerabench_u35wis62r2m.1.4_8c093924_f00.jpg) |
| `CameraBench_c3_camerabench_3m-me9axcto.1.1_60268455` / 0 | central star decoration on ceiling | 2 | `[658.0, 188.0, 812.0, 388.0]` | 0.000 | [view](frames/CameraBench_c3_camerabench_3m-me9axcto.1.1_60268455_f00.jpg) |
| `CameraBench_c3_camerabench_3m-me9axcto.1.1_60268455` / 0 | star decoration far from camera | 2 | `[656.0, 188.0, 800.0, 400.0]` | 0.000 | [view](frames/CameraBench_c3_camerabench_3m-me9axcto.1.1_60268455_f00.jpg) |
| `CameraBench_c3_camerabench_3m-me9axcto.1.1_60268455` / 0 | star decoration near camera | 2 | `[658.0, 192.0, 812.0, 400.0]` | 0.000 | [view](frames/CameraBench_c3_camerabench_3m-me9axcto.1.1_60268455_f00.jpg) |
| `CameraBench_c3_camerabench_3m-me9axcto.1.1_60268455` / 0 | star decoration on ceiling | 2 | `[584.0, 0.0, 750.0, 156.0]` | 0.000 | [view](frames/CameraBench_c3_camerabench_3m-me9axcto.1.1_60268455_f00.jpg) |
| `CameraBench_c4_camerabench_-2uia-xmjc0.5.3_877d2d07` / 0 | character | 11 | `[500.0, 175.0, 714.0, 825.0]` | 0.108 | [view](frames/CameraBench_c4_camerabench_-2uia-xmjc0.5.3_877d2d07_f00.jpg) |
| `SynFMC_c1_synfmc_rendered_traj_results_dynamic_6_video_480p_728b5ef1` / 0 | character | 1 | `[570.0, 14.0, 675.0, 408.0]` | 0.289 | [view](frames/SynFMC_c1_synfmc_rendered_traj_results_dynamic_6_video_480p_728b5ef1_f00.jpg) |
| `SynFMC_c1_synfmc_rendered_traj_results_dynamic_6_video_480p_728b5ef1` / 0 | character's back, rear of the torso | 1 | `[582.0, 100.0, 637.0, 178.0]` | 0.000 | [view](frames/SynFMC_c1_synfmc_rendered_traj_results_dynamic_6_video_480p_728b5ef1_f00.jpg) |
| `SynFMC_c1_synfmc_rendered_traj_results_dynamic_6_video_480p_728b5ef1` / 0 | character's chest, front of the torso | 1 | `[586.0, 100.0, 640.0, 166.0]` | 0.000 | [view](frames/SynFMC_c1_synfmc_rendered_traj_results_dynamic_6_video_480p_728b5ef1_f00.jpg) |
| `SynFMC_c2_synfmc_rendered_traj_results_dynamic_450_video_480p_868f34a5` / 0 | Sonic the Hedgehog | 9 | `[508.0, 327.0, 653.0, 725.0]` | 0.150 | [view](frames/SynFMC_c2_synfmc_rendered_traj_results_dynamic_450_video_480p_868f34a5_f00.jpg) |
| `SynFMC_c2_synfmc_rendered_traj_results_dynamic_450_video_480p_868f34a5` / 0 | distant hill in the background | 2 | `[0.0, 62.0, 1000.0, 362.0]` | 0.133 | [view](frames/SynFMC_c2_synfmc_rendered_traj_results_dynamic_450_video_480p_868f34a5_f00.jpg) |
| `SynFMC_c3_synfmc_rendered_traj_results_dynamic_45_video_480p_0fd58550` / 0 | background | 4 | `[0.0, 0.0, 1000.0, 600.0]` | 0.017 | [view](frames/SynFMC_c3_synfmc_rendered_traj_results_dynamic_45_video_480p_0fd58550_f00.jpg) |
| `SynFMC_c3_synfmc_rendered_traj_results_dynamic_45_video_480p_0fd58550` / 0 | character | 5 | `[12.0, 447.0, 120.0, 858.0]` | 0.000 | [view](frames/SynFMC_c3_synfmc_rendered_traj_results_dynamic_45_video_480p_0fd58550_f00.jpg) |
| `SynFMC_c3_synfmc_rendered_traj_results_dynamic_45_video_480p_0fd58550` / 0 | corner of the building on the left | 2 | `[0.0, 300.0, 100.0, 583.0]` | 0.000 | [view](frames/SynFMC_c3_synfmc_rendered_traj_results_dynamic_45_video_480p_0fd58550_f00.jpg) |
| `SynFMC_c4_synfmc_rendered_traj_results_dynamic_183_video_480p_e9b12cbc` / 0 | character | 2 | `[357.0, 262.0, 472.0, 534.0]` | 0.191 | [view](frames/SynFMC_c4_synfmc_rendered_traj_results_dynamic_183_video_480p_e9b12cbc_f00.jpg) |
| `SynFMC_c5_synfmc_rendered_traj_results_dynamic_14_video_480p_72d9da62` / 0 | man | 1 | `[400.0, 296.0, 467.0, 585.0]` | 0.592 | [view](frames/SynFMC_c5_synfmc_rendered_traj_results_dynamic_14_video_480p_72d9da62_f00.jpg) |
| `SynFMC_c5_synfmc_rendered_traj_results_dynamic_14_video_480p_72d9da62` / 0 | man's back | 1 | `[400.0, 330.0, 440.0, 420.0]` | 0.259 | [view](frames/SynFMC_c5_synfmc_rendered_traj_results_dynamic_14_video_480p_72d9da62_f00.jpg) |
| `SynFMC_c5_synfmc_rendered_traj_results_dynamic_14_video_480p_72d9da62` / 0 | man's chest | 1 | `[407.0, 343.0, 442.0, 400.0]` | 0.416 | [view](frames/SynFMC_c5_synfmc_rendered_traj_results_dynamic_14_video_480p_72d9da62_f00.jpg) |
| `internet_c0_internet_fopbvllxkz0_video-scene-00033_c8413d28` / 0 | car | 1 | `[78.0, 203.0, 800.0, 800.0]` | 0.411 | [view](frames/internet_c0_internet_fopbvllxkz0_video-scene-00033_c8413d28_f00.jpg) |
| `internet_c0_internet_fopbvllxkz0_video-scene-00033_c8413d28` / 2 | car | 1 | `[0.0, 192.0, 817.0, 856.0]` | 0.347 | [view](frames/internet_c0_internet_fopbvllxkz0_video-scene-00033_c8413d28_f02.jpg) |
| `internet_c0_internet_fopbvllxkz0_video-scene-00033_c8413d28` / 5 | car | 1 | `[0.0, 197.0, 830.0, 888.0]` | 0.324 | [view](frames/internet_c0_internet_fopbvllxkz0_video-scene-00033_c8413d28_f05.jpg) |
| `internet_c0_internet_fopbvllxkz0_video-scene-00033_c8413d28` / 10 | car | 1 | `[0.0, 192.0, 821.0, 888.0]` | 0.330 | [view](frames/internet_c0_internet_fopbvllxkz0_video-scene-00033_c8413d28_f10.jpg) |
| `internet_c0_internet_fopbvllxkz0_video-scene-00033_c8413d28` / 15 | car | 1 | `[0.0, 250.0, 856.0, 888.0]` | 0.282 | [view](frames/internet_c0_internet_fopbvllxkz0_video-scene-00033_c8413d28_f15.jpg) |
| `internet_c2_internet_2d0iwohbsju_video-scene-00020_fdbaa511` / 0 | ornate ceiling corner | 2 | `[320.0, 0.0, 700.0, 420.0]` | 0.044 | [view](frames/internet_c2_internet_2d0iwohbsju_video-scene-00020_fdbaa511_f00.jpg) |
| `internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874` / 0 | a different tree in the background | 2 | `null` | — | [view](frames/internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874_f00.jpg) |
| `internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874` / 0 | a rock in the background | 2 | `[885.0, 288.0, 1000.0, 600.0]` | 0.000 | [view](frames/internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874_f00.jpg) |
| `internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874` / 0 | a tree in the background | 2 | `null` | — | [view](frames/internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874_f00.jpg) |
| `internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874` / 0 | a tree in the center of the frame | 1 | `null` | — | [view](frames/internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874_f00.jpg) |
| `internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874` / 0 | a tree on the left side of the frame | 2 | `null` | — | [view](frames/internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874_f00.jpg) |
| `internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874` / 0 | a tree on the right side of the frame | 2 | `null` | — | [view](frames/internet_c3_internet_4cz3oqlfupm_video-scene-00023_6ef46874_f00.jpg) |
| `internet_c4_internet_an2sz4h4qzk_video-scene-00048_265c8450` / 0 | horse | 2 | `[500.0, 575.0, 550.0, 825.0]` | 0.000 | [view](frames/internet_c4_internet_an2sz4h4qzk_video-scene-00048_265c8450_f00.jpg) |
| `internet_c5_internet_0knwuip85a8_video-scene-00009_39b26678` / 0 | police car | 5 | `[357.0, 458.0, 447.0, 538.0]` | 0.149 | [view](frames/internet_c5_internet_0knwuip85a8_video-scene-00009_39b26678_f00.jpg) |
| `k700_c0_k700_paragliding_95df0yxigbw_000076_000086_video-scene-00000_a1d4095b` / 0 | paraglider | 1 | `[470.0, 620.0, 585.0, 942.0]` | 0.052 | [view](frames/k700_c0_k700_paragliding_95df0yxigbw_000076_000086_video-scene-00000_a1d4095b_f00.jpg) |
| `k700_c0_k700_paragliding_95df0yxigbw_000076_000086_video-scene-00000_a1d4095b` / 0 | paraglider's back | 1 | `[505.0, 656.0, 547.0, 752.0]` | 0.000 | [view](frames/k700_c0_k700_paragliding_95df0yxigbw_000076_000086_video-scene-00000_a1d4095b_f00.jpg) |
| `k700_c0_k700_paragliding_95df0yxigbw_000076_000086_video-scene-00000_a1d4095b` / 0 | paraglider's chest | 1 | `[507.0, 657.0, 545.0, 728.0]` | 0.000 | [view](frames/k700_c0_k700_paragliding_95df0yxigbw_000076_000086_video-scene-00000_a1d4095b_f00.jpg) |
| `k700_c1_700_playing_polo_j7j8_hz_hx8_000018_000028_video-scene-00001_68f048f4` / 0 | horse in the foreground | 1 | `[367.0, 456.0, 388.0, 535.0]` | 0.018 | [view](frames/k700_c1_700_playing_polo_j7j8_hz_hx8_000018_000028_video-scene-00001_68f048f4_f00.jpg) |
| `k700_c1_700_playing_polo_j7j8_hz_hx8_000018_000028_video-scene-00001_68f048f4` / 0 | horse's back, rear of the torso | 1 | `[370.0, 468.0, 387.0, 510.0]` | 0.000 | [view](frames/k700_c1_700_playing_polo_j7j8_hz_hx8_000018_000028_video-scene-00001_68f048f4_f00.jpg) |
| `k700_c1_700_playing_polo_j7j8_hz_hx8_000018_000028_video-scene-00001_68f048f4` / 0 | horse's chest, front of the torso | 1 | `[372.0, 470.0, 387.0, 500.0]` | 0.000 | [view](frames/k700_c1_700_playing_polo_j7j8_hz_hx8_000018_000028_video-scene-00001_68f048f4_f00.jpg) |
| `k700_c3_k700_skiing_mono_rqfndqdk5ny_000080_000090_video-scene-00000_fd807178` / 0 | a tree in the distance | 2 | `[0.0, 0.0, 1000.0, 50.0]` | 0.000 | [view](frames/k700_c3_k700_skiing_mono_rqfndqdk5ny_000080_000090_video-scene-00000_fd807178_f00.jpg) |
| `k700_c3_k700_skiing_mono_rqfndqdk5ny_000080_000090_video-scene-00000_fd807178` / 0 | the sled | 2 | `[0.0, 0.0, 707.0, 1000.0]` | 0.707 | [view](frames/k700_c3_k700_skiing_mono_rqfndqdk5ny_000080_000090_video-scene-00000_fd807178_f00.jpg) |
| `k700_c4_k700_jogging_qy8rjbxblna_000116_000126_video-scene-00001_ab85ad6e` / 0 | man | 2 | `[648.0, 227.0, 870.0, 997.0]` | 0.084 | [view](frames/k700_c4_k700_jogging_qy8rjbxblna_000116_000126_video-scene-00001_ab85ad6e_f00.jpg) |
| `k700_c5_700_snowboarding_9o2tvonlrmu_000003_000013_video-scene-00000_227e207a` / 0 | skier | 1 | `[398.0, 267.0, 588.0, 704.0]` | 0.423 | [view](frames/k700_c5_700_snowboarding_9o2tvonlrmu_000003_000013_video-scene-00000_227e207a_f00.jpg) |
| `k700_c5_700_snowboarding_9o2tvonlrmu_000003_000013_video-scene-00000_227e207a` / 0 | skier's back | 1 | `[483.0, 340.0, 557.0, 557.0]` | 0.000 | [view](frames/k700_c5_700_snowboarding_9o2tvonlrmu_000003_000013_video-scene-00000_227e207a_f00.jpg) |
| `k700_c5_700_snowboarding_9o2tvonlrmu_000003_000013_video-scene-00000_227e207a` / 0 | skier's chest | 1 | `[487.0, 333.0, 555.0, 440.0]` | 0.000 | [view](frames/k700_c5_700_snowboarding_9o2tvonlrmu_000003_000013_video-scene-00000_227e207a_f00.jpg) |
| `llava178k_c0_videos_youtube_video_2024_ytb_djaut-hx2iu_video-scene-00000_bf9d0572` / 0 | background pole | 1 | `null` | — | [view](frames/llava178k_c0_videos_youtube_video_2024_ytb_djaut-hx2iu_video-scene-00000_bf9d0572_f00.jpg) |
| `llava178k_c0_videos_youtube_video_2024_ytb_djaut-hx2iu_video-scene-00000_bf9d0572` / 0 | car | 1 | `[0.0, 106.0, 1000.0, 712.0]` | 0.712 | [view](frames/llava178k_c0_videos_youtube_video_2024_ytb_djaut-hx2iu_video-scene-00000_bf9d0572_f00.jpg) |
| `llava178k_c1_videos_youtube_video_2024_ytb_pzg1hdiwfyk_video-scene-00002_fb3c4084` / 0 | man | 2 | `[312.0, 452.0, 788.0, 998.0]` | 0.208 | [view](frames/llava178k_c1_videos_youtube_video_2024_ytb_pzg1hdiwfyk_video-scene-00002_fb3c4084_f00.jpg) |
| `llava178k_c2_videos_youtube_video_2024_ytb_opt9l_7g68m_video-scene-00004_11dfd093` / 0 | corner of the castle | 2 | `[175.0, 288.0, 300.0, 388.0]` | 0.009 | [view](frames/llava178k_c2_videos_youtube_video_2024_ytb_opt9l_7g68m_video-scene-00004_11dfd093_f00.jpg) |
| `llava178k_c4_videos_youtube_video_2024_ytb_pcj0xjv7gi0_video-scene-00002_b63a5bf6` / 0 | red car | 13 | `[350.0, 487.0, 615.0, 552.0]` | 0.158 | [view](frames/llava178k_c4_videos_youtube_video_2024_ytb_pcj0xjv7gi0_video-scene-00002_b63a5bf6_f00.jpg) |
| `llava178k_c5_videos_youtube_video_2024_ytb_16mzjk4aojs_video-scene-00000_b36d2eb1` / 0 | back of harvester | 1 | `[388.0, 202.0, 875.0, 375.0]` | 0.010 | [view](frames/llava178k_c5_videos_youtube_video_2024_ytb_16mzjk4aojs_video-scene-00000_b36d2eb1_f00.jpg) |
| `llava178k_c5_videos_youtube_video_2024_ytb_16mzjk4aojs_video-scene-00000_b36d2eb1` / 0 | front of harvester | 1 | `[0.0, 467.0, 220.0, 815.0]` | 0.000 | [view](frames/llava178k_c5_videos_youtube_video_2024_ytb_16mzjk4aojs_video-scene-00000_b36d2eb1_f00.jpg) |
| `llava178k_c5_videos_youtube_video_2024_ytb_16mzjk4aojs_video-scene-00000_b36d2eb1` / 0 | harvester | 1 | `[0.0, 202.0, 1000.0, 812.0]` | 0.033 | [view](frames/llava178k_c5_videos_youtube_video_2024_ytb_16mzjk4aojs_video-scene-00000_b36d2eb1_f00.jpg) |

Raw model replies and parsed boxes are in `detections/`; `requests.json` maps
the unique results back to every original trace step.
