- First call `camera_motion`: if its label is 'static', the camera is fixed -- the speed table then shows
  estimation jitter, not motion. Only then read `camera_profile` for the rhythm.
- Call `camera_profile`; read speed per span in time order (still -> burst, steady, slowing to a stop, speeding up).
- `camera_motion` tells you which motion it is, if the options also name one.
