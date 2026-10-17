"""One dispatch boundary, with host-derived question intervals and perception failures."""
import re

from .reports import Report
from .workspace import CandidateRequired, WorkspaceError


def question_window(question, duration):
    """Only explicit seconds in the stem narrow measurements; model calls cannot sample them."""
    number = r'(\d+(?:\.\d+)?)'
    unit = r'(?:s\b|seconds?\b)'
    match = re.search(r'(?:between|from)\s+' + number + r'\s*' + unit +
                      r'?\s*(?:and|to|–|-)\s*' + number + r'\s*' + unit, question, re.I)
    if match:
        a, b = map(float, match.groups())
        return (a, b) if a <= b else None
    match = re.search(r'(?:first|initial)\s+' + number + r'\s*' + unit, question, re.I)
    if match:
        return 0.0, float(match[1])
    match = re.search(r'(?:last|final)\s+' + number + r'\s*' + unit, question, re.I)
    if match and duration is not None:
        return max(0.0, duration - float(match[1])), duration
    return None


class Toolset:
    def __init__(self, workspace, qwen, question, options, caption):
        self.workspace = workspace
        self.qwen = qwen
        self.question = question
        self.options = options
        self.caption = caption
        duration = (workspace.clip.length - 1) / workspace.clip.fps if workspace.clip.fps else None
        self.window = question_window(question, duration)

    def call(self, tool, **arguments):
        from .tools import camera, entity, orientation, group, frames, inventory, ask_video
        ws = self.workspace
        try:
            if tool in ('track_camera', 'track_camera_zoom', 'check_scene_revisit'):
                return getattr(camera, tool)(ws, window=self.window)
            if tool == 'track_entity':
                return entity.track_entity(ws, qwen=self.qwen, window=self.window, **arguments)
            if tool == 'track_orientation':
                return orientation.track_orientation(ws, qwen=self.qwen, window=self.window, **arguments)
            if tool == 'track_multiple_entities':
                return group.track_multiple_entities(ws, qwen=self.qwen, window=self.window, **arguments)
            if tool == 'get_frames':
                return frames.get_frames(ws, **arguments)
            if tool == 'get_segmentation_mask':
                return inventory.get_segmentation_mask(ws, **arguments)
            if tool == 'detect_all_entities':
                return inventory.detect_all_entities(ws, self.qwen, self.question, self.options, self.caption)
            if tool == 'choose_segmentation_candidate':
                return inventory.choose_segmentation_candidate(ws, **arguments)
            if tool == 'ask_video':
                return ask_video.ask_video(ws, self.qwen, **arguments)
            raise ValueError(f'Unknown tool: {tool}')
        except CandidateRequired as error:
            return inventory.candidate_report(ws, error.selection)
        except WorkspaceError as error:
            return Report.unavailable(tool, str(error))
