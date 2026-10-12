- Call `camera_profile` for motion presence, tempo and jitter. Small net displacement can coexist with
  an out-and-return movement. If the presence and tempo summaries disagree, inspect the spans and frames;
  neither label automatically vetoes the other.
- Call `camera_profile`; read speed per span in time order (still -> burst, steady, slowing to a stop, speeding up).
- `camera_motion` tells you which motion it is, if the options also name one.
