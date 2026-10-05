- Call `elevation`: + = the camera is above the subject (looking down = high angle), - = below (low angle),
  near 0 = eye level.
- Without a found ground plane, gravity is guessed from the camera; check the frames then.
- 'eye' (within +-10 deg) is eye level even if the median is a few degrees up or down; 'changing' needs the
  angle to move between bands during the clip.
