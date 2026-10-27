"""Scripted local-Qwen replays of the six worked planning examples."""
import json
import pytest
from PIL import Image

from .agent import AgentV6
from .context import Conversation
from .reports import Report
from .test_calls import Scripted
from .test_workspace import fixture_workspace
from .toolset import question_window


class RecordingTools:
    def __init__(self):
        self.calls = []

    def call(self, tool, **arguments):
        self.calls.append((tool, arguments))
        if tool == 'track_entity' and arguments['main'] == 'bus':
            return Report.unavailable('bus motion', 'No object found. Suggest a singular descriptive alternative; absence is not established.')
        return Report(tool, 'Measured evidence', reading_full='Full reading contract', reading_short='Short notes')


EXAMPLES = [
    ('How many fish are swimming to the right?', ['fish'],
     '1. [counting] how many fish, and when visible\n2. [entity motion] each fish direction in the camera view\n3. [counting] check right-movers',
     [('get_segmentation_mask', {'entities':['fish'], 'times':[0,1]}),
      ('track_multiple_entities', {'category':'fish','frame_of_reference':'camera'}),
      ('get_segmentation_mask', {'entities':['fish'], 'times':[2,3]})]),
    ('Relative to its starting orientation, how is the observer moving?', [],
     '1. [camera motion] camera translation and turning', [('track_camera',{})]),
    ('Does the camera pan left before the skier jumps?', ['skier'],
     '1. [visual] when the skier jumps\n2. [camera motion] when the left pan starts\n3. compare',
     [('ask_video',{'question':'At which time does the skier jump?'}), ('track_camera',{})]),
    ('Does the dog move closer to the fence?', ['dog','fence'],
     '1. [entities and landmarks] dog distance from fence',
     [('track_entity',{'main':'dog','others':['fence'],'frame_of_reference':'subject'})]),
    ('What direction is the bus moving?', ['bus'],
     '1. [entity motion] whether a bus appears\n2. [entity motion] its direction',
     [('track_entity',{'main':'bus'}), ('ask_video',{'question':'Is a bus visible anywhere in the video?'})]),
    ("How is the people's location moving relative to his/her/its starting orientation?", ['person'],
     '1. [entity motion] person motion and turning in own starting facing',
     [('track_entity',{'main':'person','frame_of_reference':'subject'})]),
]


@pytest.mark.parametrize('question,entities,plan,calls', EXAMPLES)
def test_worked_examples(question, entities, plan, calls):
    ws, *_ = fixture_workspace()
    # Eight automatic keyframes, retained through the same answer conversation.
    ws.clip.record['sampled16'] = list(range(0,31,2))
    qwen = Scripted(json.dumps({'entities':entities}), 'This question is about measured quantities.\n' + plan,
                    *[json.dumps({'tool':name,'arguments':args}) for name,args in calls],
                    '{"answer":true}', '{"letter":"B","because":"Measured evidence decides."}')
    dispatcher = RecordingTools()
    agent = AgentV6(qwen=qwen, dispatcher=dispatcher, captioner=lambda ws,q: 'An outdoor scene.')
    result = agent.solve({'id':'q','question':question,'options':{'A':'first option','B':'second option'}}, workspace=ws)
    assert result['letter'] == 'B'
    assert [name for name,_ in dispatcher.calls] == [name for name,_ in calls]
    assert result['tool_calls'] == len(calls)
    assert result['answer'].startswith('B: second option —')
    first, last = qwen.calls[0], qwen.calls[-1]
    image_count = lambda messages: sum(part['type'] == 'image' for m in messages for part in m['content'] if isinstance(m['content'], list))
    assert image_count(first) == image_count(last) == 8
    assert 'Full reading contract' in str(last)
    assert 'No tools in this turn' in str(last[-1])
    assert 'track_camera' not in str(first[-1])


def test_trimming_only_removes_requested_images_oldest_first():
    image = Image.new('RGB',(448,448))
    context = Conversation('system', 2000)
    context.add('user','question',images=[image]*8,kind='opening')
    context.add('assistant','1. plan',kind='plan')
    context.add('user','first evidence',kind='report')
    old = context.add('user','old request',images=[image]*3,kind='requested')
    new = context.add('user','new request',images=[image],kind='requested')
    context.fit()
    assert not old.images and new.images
    assert len(context.turns[1].images) == 8
    assert context.turns[2].text == '1. plan' and context.turns[3].text == 'first evidence'


def test_budget_warns_on_calls_twelve_thirteen_fourteen():
    ws, *_ = fixture_workspace()
    qwen = Scripted('{"entities":[]}', 'This question is about camera motion.\n1. [camera motion] rotation',
                    *(['{"tool":"track_camera","arguments":{}}'] * 14), '{"letter":"B","because":"turning"}')
    tools = RecordingTools()
    result = AgentV6(qwen=qwen, dispatcher=tools, captioner=lambda *args:'scene').solve(
        {'id':'q','question':'How does the camera move?','options':{'A':'still','B':'turning'}}, workspace=ws)
    assert result['tool_calls'] == len(tools.calls) == 14
    final = str(qwen.calls[-1])
    for count in (12,13,14):
        assert f'WARNING: call {count}/14 used' in final
    assert result['process']['planned_items'] == 1


def test_corrected_call_uses_one_budget_slot():
    ws, *_ = fixture_workspace()
    qwen = Scripted('{"entities":["runner"]}', 'This question is about motion.\n1. [entity motion] direction',
                    '{"tool":"track_entity","arguments":{"main":"runner","frame":"subj"}}',
                    '{"answer":true}', '{"letter":"B","because":"forward"}')
    result = AgentV6(qwen=qwen, dispatcher=RecordingTools(), captioner=lambda *args:'scene').solve(
        {'id':'q','question':'Direction?','options':{'A':'back','B':'forward'}}, workspace=ws)
    assert result['tool_calls'] == 1 and result['process']['normalizer_fixes'] == 1


def test_time_windows_come_only_from_explicit_question_seconds():
    assert question_window('What happens in the first 2 seconds?',10) == (0,2)
    assert question_window('In the last 2 seconds?',10) == (8,10)
    assert question_window('Between 1 and 3 seconds, where?',10) == (1,3)
    assert question_window('At generator frame 31, where?',10) is None
    assert question_window('How does it move?',10) is None


def test_generation_reserve_trims_requested_images_before_rejecting():
    context = Conversation('system', 1000)
    image = Image.new('RGB',(640,640))
    requested = context.add('user','frame',images=[image],kind='requested')
    context.add('user','x' * 500,kind='report')
    assert context.tokens() < 1000
    assert context.fit(reserve=600) <= 400
    assert not requested.images


def test_context_pressure_removes_old_control_turns_and_protects_evidence_and_schemas():
    context = Conversation('system', 2500)
    opening = context.add('user', 'question', images=[Image.new('RGB', (448, 448))] * 8, kind='opening')
    plan = context.add('assistant', '1. Keep this plan.', kind='plan')
    schema_menu = context.add('user', 'Full tool schemas.', kind='act')
    call = context.add('assistant', '{"tool":"track_camera","arguments":{}}', kind='act')
    report = context.add('user', 'Keep every measurement. ' * 20, kind='report')
    obsolete = context.add('user', 'Repeated tool menu and reminder. ' * 150, kind='act')
    context.add('assistant', '{"tool":"track_camera","arguments":{}}', kind='act')
    revised = context.add('assistant', 'Revised plan:\n1. Keep this revision.', kind='plan')
    current = context.add('user', 'Current instructions.', kind='answer')
    assert context.fit(reserve=512) + 512 <= context.budget
    assert obsolete.trimmed and not call.trimmed
    assert len(opening.images) == 8
    assert plan.text == '1. Keep this plan.' and revised.text.startswith('Revised plan:')
    assert report.text == 'Keep every measurement. ' * 20
    assert schema_menu.text == 'Full tool schemas.' and current.text == 'Current instructions.'


def test_context_pressure_can_remove_old_call_replies_but_not_plan_revisions():
    context = Conversation('system', 600)
    context.add('user', 'Full tool schemas.', kind='act')
    old_call = context.add('assistant', 'malformed call ' * 200, kind='act')
    revised = context.add('assistant', 'Revised plan:\n1. Keep this revision.', kind='plan')
    report = context.add('user', 'Measured evidence.', kind='report')
    context.add('user', 'Answer now.', kind='answer')
    assert context.fit(reserve=100) <= 500
    assert old_call.trimmed and revised.text.startswith('Revised plan:')
    assert report.text == 'Measured evidence.'


def test_context_pressure_answers_before_another_act_and_keeps_all_reports():
    class MeasuringQwen:
        def __init__(self):
            self.calls = []

        def generate(self, messages, **kwargs):
            self.calls.append(messages)
            instruction = messages[-1]['content'][-1]['text']
            if 'No tools in this turn' in instruction:
                return '{"letter":"B","because":"measured"}'
            if len(self.calls) == 1:
                return '{"entities":[]}'
            if len(self.calls) == 2:
                return '1. [camera motion] Measure camera motion.'
            return '{"tool":"track_camera","arguments":{}}'

    class LargeReports:
        def __init__(self):
            self.calls = 0

        def call(self, *args, **kwargs):
            self.calls += 1
            return Report('motion', f'Moves right; observation {self.calls}.', [('Measured distance 2 m.',)] * 150,
                          columns=('Measurement',), reading_full='Full measurement contract.')

    ws, *_ = fixture_workspace()
    qwen = MeasuringQwen()
    ws.clip.record['sampled16'] = list(range(0, 31, 2))
    result = AgentV6(qwen=qwen, token_budget=10000, dispatcher=LargeReports(),
                     captioner=lambda *a: 'scene').solve(
        {'id': 'q', 'question': 'Direction?', 'options': {'A': 'left', 'B': 'right'}}, workspace=ws)
    assert result['letter'] == 'B' and 0 < result['tool_calls'] < 14
    assert any(step['phase'] == 'context_budget' for step in result['trace'])
    final = '\n'.join(part['text'] for m in qwen.calls[-1] for part in m['content'] if part['type'] == 'text')
    assert 'Full measurement contract.' in final
    assert all(report['rendered'] in final for report in result['reports'])
    assert sum(p['type'] == 'image' for m in qwen.calls[-1] for p in m['content']) == 8
    for messages in qwen.calls:
        context = Conversation('', 10000)
        context.turns = []
        for message in messages:
            text = message['content'][-1]['text']
            images = [part['image'] for part in message['content'] if part['type'] == 'image']
            context.add(message['role'], text, images=images)
        assert context.tokens() + 512 <= 10000


def test_repeated_report_readings_keep_one_complete_copy_under_context_pressure():
    context = Conversation('system', 900)
    reading = 'Measured distance fell from 5 m to 2 m. ' * 50
    original = context.add('user', reading, kind='report')
    duplicates = [context.add('user', reading, kind='report') for _ in range(4)]
    context.add('user', 'Answer now.', kind='answer')
    assert context.fit(reserve=100) <= 800
    assert original.text == reading and not original.trimmed
    assert all(t.summary == reading and 'Same report as turn 1' in t.text for t in duplicates)
    assert context.fit(reserve=100) <= 800
    assert original.text == reading


def test_unique_evidence_that_exceeds_final_context_is_explicitly_unanswered_and_retained():
    from .evaluation.campaign import check_records
    from .evaluation.metrics import accuracy

    class OversizedReport:
        def call(self, *args, **kwargs):
            return Report('motion', 'Measured evidence.', reading_full='Unique measured evidence. ' * 5000)

    ws, *_ = fixture_workspace()
    qwen = Scripted('{"entities":[]}', '1. Measure motion.', '{"tool":"track_camera","arguments":{}}')
    result = AgentV6(qwen=qwen, token_budget=10000, dispatcher=OversizedReport(),
                     captioner=lambda *a: 'scene').solve(
        {'id': 'q', 'question': 'Direction?', 'options': {'A': 'left', 'B': 'right'}}, workspace=ws)
    assert result['status'] == 'unanswered' and result['letter'] is None
    assert not result['answered'] and 'error' not in result
    assert result['context_budget_stop']['tokens'] > 10000
    assert result['peak_context_tokens'] + 512 <= 10000
    assert len(qwen.calls) == 3 and result['process']['doctor_calls'] == 0
    assert ('Unique measured evidence. ' * 5000).strip() in result['reports'][0]['rendered']
    assert result['plan'] == '1. Measure motion.'
    assert any(t['phase'] == 'context_budget' for t in result['trace'])
    check_records({'q': result})
    assert accuracy([{**result, 'reference': {'answer': 'B'}}])['accuracy'] == 0


def test_auxiliary_calls_do_not_advance_plan_without_explicit_completion():
    ws, *_ = fixture_workspace()
    qwen = Scripted('{"entities":["car"]}',
                    'This question is about motion.\n1. [entity motion] car direction\n2. [camera motion] camera direction',
                    '{"tool":"get_frames","arguments":{}}',
                    '{"tool":"track_camera","arguments":{}}',
                    '{"answer":true,"completed_plan_items":[2]}',
                    '{"letter":"B","because":"evidence"}')
    result = AgentV6(qwen=qwen, dispatcher=RecordingTools(), captioner=lambda *a:'scene').solve(
        {'id':'q','question':'Direction?','options':{'A':'left','B':'right'}},workspace=ws)
    assert 'Plan item 1 of 2 next' in str(qwen.calls[3][-1])
    assert result['process']['executed_items'] == 1


def test_plan_revision_resets_completion_and_retains_prior_plan_in_context():
    ws, *_ = fixture_workspace()
    qwen = Scripted('{"entities":[]}',
                    'This question is about motion.\n1. [camera motion] translation\n2. [camera motion] rotation',
                    '{"tool":"track_camera","arguments":{},"completed_plan_items":[1,2]}',
                    'Revised plan:\n1. [visual] inspect scene', '{"answer":true}',
                    '{"letter":"B","because":"visible evidence"}')
    result = AgentV6(qwen=qwen, dispatcher=RecordingTools(), captioner=lambda *a:'scene').solve(
        {'id':'q','question':'Direction?','options':{'A':'left','B':'right'}},workspace=ws)
    assert result['process']['executed_items'] == 0
    assert result['process']['planned_items'] == 1
    assert 'translation' in str(qwen.calls[-1])


@pytest.mark.parametrize('declaration', [
    '{"answer":"B"}', '{"final_answer":"B"}', '{"Letter":"B"}',
    '{"action":"answer"}', '{"tool":"answer"}', '{"answer": B}',
    'The answer is B.', 'B: right', 'I can now answer.',
])
def test_act_answer_declarations_go_to_answer_instructions_without_a_doctor(declaration):
    ws, *_ = fixture_workspace()
    qwen = Scripted('{"entities":[]}', '1. Inspect the visible evidence.', declaration,
                    '{"letter":"B","because":"measured"}')
    tools = RecordingTools()
    result = AgentV6(qwen=qwen, dispatcher=tools, captioner=lambda *a: 'scene').solve(
        {'id': 'q', 'question': 'Direction?', 'options': {'A': 'left', 'B': 'right'}}, workspace=ws)
    assert result['letter'] == 'B'
    assert result['process']['doctor_calls'] == 0 and not tools.calls
    assert len(qwen.calls) == 4
    assert 'No tools in this turn' in str(qwen.calls[-1][-1])


@pytest.mark.parametrize('first', [
    Report.unavailable('runner motion', 'No usable positions.'),
    Report('Choose a segmentation candidate', 'Choose red or blue.', status='partial',
           reason='An instance choice is needed before measuring.', images=[Image.new('RGB', (16, 16))]),
])
def test_full_contract_is_delivered_on_first_measurement_after_failed_or_candidate_report(first):
    class Reports:
        def __init__(self):
            self.replies = iter([first, Report('motion', 'Moves right.', reading_full='Full motion contract.',
                                               reading_short='Short motion notes.'),
                                 Report('motion', 'Moves right.', reading_full='Full motion contract.',
                                        reading_short='Short motion notes.')])

        def call(self, *args, **kwargs):
            return next(self.replies)

    ws, *_ = fixture_workspace()
    qwen = Scripted('{"entities":["runner"]}', '1. Measure runner motion.',
                    *(['{"tool":"track_entity","arguments":{"main":"runner"}}'] * 3),
                    '{"answer":true}', '{"letter":"B","because":"measured"}')
    result = AgentV6(qwen=qwen, dispatcher=Reports(), captioner=lambda *a: 'scene').solve(
        {'id': 'q', 'question': 'Direction?', 'options': {'A': 'left', 'B': 'right'}}, workspace=ws)
    assert 'Full motion contract.' in result['reports'][1]['rendered']
    assert 'Short motion notes.' in result['reports'][2]['rendered']
    assert sum('Full motion contract.' in report['rendered'] for report in result['reports']) == 1
    assert 'Full motion contract.' in str(qwen.calls[-1])


def test_doubled_braces_in_act_are_recoverable_and_doctor_guidance_is_visible():
    ws, *_ = fixture_workspace()
    advice = 'To the agent: Remove the extra outer braces.'
    qwen = Scripted('{"entities":[]}', '1. Measure the camera.',
                    '{{"tool":"track_camera","arguments":{}}}', advice,
                    '{"tool":"track_camera","arguments":{}}', '{"answer":true}',
                    '{"letter":"B","because":"measured"}')
    result = AgentV6(qwen=qwen, dispatcher=RecordingTools(), captioner=lambda *a: 'scene').solve(
        {'id': 'q', 'question': 'Direction?', 'options': {'A': 'left', 'B': 'right'}}, workspace=ws)
    assert result['letter'] == 'B' and result['tool_calls'] == 1
    assert result['process']['doctor_calls'] == 1
    assert advice in str(qwen.calls[-1])
