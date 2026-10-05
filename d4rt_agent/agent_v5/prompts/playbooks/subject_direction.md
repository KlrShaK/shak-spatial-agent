- Call `entity_track` on the subject (frame 'scene' by default). If the question says "relative to the camera",
  use frame 'camera'; for "in the picture/frame" wording use 'image'.
- Left/right in the 'scene' frame is reliable; forward/backward from depth is close to a guess -- decide that part
  from the frames (does it get bigger and lower in the picture = toward the camera).
- Describe the path in time order (chronological rule).
- If the track is short or odd, `show` the subject to check the right thing was segmented.
