"""Small public tool contracts. All times are seconds or labels of shown frames."""

DESCRIPTIONS = {
    'track_camera': 'Camera translation and pan/tilt/roll; use for observer motion, not entity motion.',
    'track_entity': 'Entity motion and relative positions; world by default, subject for its starting facing, camera for the moving view.',
    'track_orientation': 'Facing relative to camera, own start, or another named entity; not translation.',
    'track_multiple_entities': 'Each category instance and counts by direction; you decide the final count.',
    'track_camera_zoom': 'Focal change and forward translation, separating zoom from dolly.',
    'check_scene_revisit': 'Geometric overlap between the start and end views.',
    'detect_all_entities': 'Discover nouns and segmented instances on demand; not proof of absence.',
    'get_segmentation_mask': 'Visible spans, size phases, edges and optional masked frames for named entities.',
    'get_frames': 'Four plain frames at up to four times or over a time range; no measurements.',
    'ask_video': 'Ask a separate video viewer about visual facts or events; may include question and options.',
    'choose_segmentation_candidate': 'Choose a candidate colour immediately after an ambiguity image.',
}

# A tuple is an enum; the other types name deliberately small validators.
FIELDS = {
    'track_camera': {}, 'track_camera_zoom': {}, 'check_scene_revisit': {}, 'detect_all_entities': {},
    'track_entity': {'main': 'entity', 'others': 'entities', 'frame_of_reference': ('world', 'camera', 'subject')},
    'track_orientation': {'main': 'entity', 'frame_of_reference': 'orientation_reference'},
    'track_multiple_entities': {'category': 'entity', 'frame_of_reference': ('world', 'camera', 'own')},
    'get_segmentation_mask': {'entities': 'entities', 'times': 'times8'},
    'get_frames': {'times': 'times4', 'range': 'range'},
    'ask_video': {'question': 'text'},
    'choose_segmentation_candidate': {'colour': 'text'},
}
REQUIRED = {'track_entity': ('main',), 'track_orientation': ('main',),
            'track_multiple_entities': ('category',), 'get_segmentation_mask': ('entities',),
            'ask_video': ('question',), 'choose_segmentation_candidate': ('colour',)}
DEFAULTS = {'track_entity': {'others': [], 'frame_of_reference': 'world'},
            'track_orientation': {'frame_of_reference': 'camera'},
            'track_multiple_entities': {'frame_of_reference': 'world'}}
EXAMPLES = {
    'track_entity': [{'main': 'runner'}, {'main': 'dog', 'others': ['fence'], 'frame_of_reference': 'subject'}],
    'track_orientation': [{'main': 'bear', 'frame_of_reference': 'camera'}, {'main': 'bear', 'frame_of_reference': 'own'}],
    'track_multiple_entities': [{'category': 'fish'}, {'category': 'bear', 'frame_of_reference': 'own'}],
    'get_segmentation_mask': [{'entities': ['fish']}, {'entities': ['runner'], 'times': [0.0, 1.0]}],
    'get_frames': [{'times': [0.0, 1.0]}, {'range': [0.0, 2.0]}],
    'ask_video': [{'question': 'When does the skier jump?'}, {'question': 'What colour is the bus?'}],
    'choose_segmentation_candidate': [{'colour': 'red'}, {'colour': 'blue'}],
}


def schema(tool):
    return {'tool': tool, 'when': DESCRIPTIONS[tool], 'arguments': FIELDS[tool],
            'required': REQUIRED.get(tool, ()), 'defaults': DEFAULTS.get(tool, {}),
            'examples': [{'tool': tool, 'arguments': values} for values in EXAMPLES.get(tool, [{}, {}])]}


def visible_tools(pending=False):
    return [name for name in FIELDS if pending or name != 'choose_segmentation_candidate']
