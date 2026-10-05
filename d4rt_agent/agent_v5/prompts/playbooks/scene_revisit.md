- Call `scene_revisit`: the share of the first view seen again at the end.
- A large pan or translation (in `camera_motion`) with a low shared share means the view changed; a camera that
  turns away and back can end on the same view.
- This kind answers 'is the background / scene at the end the same as at the start?', including 'same / slight
  shift / different' options.
