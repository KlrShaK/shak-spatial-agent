import pytest
from .calls import CallProcessor, normalize, recover_answer, validate


class Scripted:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return next(self.responses)


def test_normalizer_repairs_names_aliases_and_python_json():
    call, fixes = normalize("```json\n{'tool':'track_entity','arguments':{'main':'the runner','frame':'subj',},}\n```", ['runner'])
    assert call['arguments'] == dict(main='runner', frame_of_reference='subject', others=[])
    assert fixes and not validate(call)


@pytest.mark.parametrize('raw,fragment', [
    ({'tool':'track_entity','arguments':{'main':'c1#0'}}, 'names in words'),
    ({'tool':'get_frames','arguments':{'frames':[0,31]}}, 'generator frames'),
    ({'tool':'get_frames','arguments':{'inspection':'inspection_2'}}, 'inspection handles'),
    ({'tool':'track_entity','arguments':{'main':'cam'}}, 'track_camera'),
])
def test_legacy_refusals_give_actionable_messages(raw, fragment):
    result = CallProcessor().process(raw)
    assert result.call is None
    assert fragment.lower() in result.message.lower()


def test_plain_name_is_accepted_without_id():
    result = CallProcessor().process({'tool':'track_entity','arguments':{'main':'snowboarder'}})
    assert result.call['arguments']['main'] == 'snowboarder'


def test_doctor_cannot_change_valid_value():
    doctor = Scripted('{"tool":"track_entity","arguments":{"main":"car","frame_of_reference":"world"}}')
    processor = CallProcessor(doctor)
    result = processor.process({'tool':'track_entity','arguments':{'main':'runner','frame_of_reference':'sideways'}})
    assert result.call is None
    assert 'changed valid main' in result.message
    assert processor.doctor_calls == 1


def test_doctor_fills_invalid_value_once():
    doctor = Scripted('{"tool":"track_entity","arguments":{"main":"runner","frame_of_reference":"world"}}')
    processor = CallProcessor(doctor)
    result = processor.process({'tool':'track_entity','arguments':{'main':'runner','frame_of_reference':'bad'}})
    assert result.call['arguments']['main'] == 'runner'
    assert processor.counts['doctor_fixes'] == 1
    assert len(doctor.calls) == 1


def test_doctor_limit_and_candidate_visibility():
    processor = CallProcessor(Scripted(*(['To the agent: name an entity'] * 5)))
    for _ in range(8):
        assert processor.process({'tool':'track_entity','arguments':{}}).call is None
    assert processor.doctor_calls == 5
    call = {'tool':'choose_segmentation_candidate','arguments':{'colour':'red'}}
    assert validate(call) and not validate(call, pending=True)


@pytest.mark.parametrize('raw,letter', [
    ('{"letter":"B","because":"the distance decreases"}', 'B'),
    ('B: closer — the distance decreases', 'B'),
    ('The answer is B.', 'B'),
    ('Option A or option B.', None),
    ('It is getting closer.', None),
])
def test_answer_recovery(raw, letter):
    result = recover_answer(raw, {'A':'away','B':'closer'})
    assert (result['letter'] if result else None) == letter


def test_answer_doctor_cannot_invent_letter():
    doctor = Scripted('{"letter":"B","because":"closer"}')
    assert recover_answer('The distance decreases.', {'A':'away','B':'closer'}, doctor) is None


def test_answer_doctor_shares_call_budget():
    doctor = CallProcessor(Scripted('{"letter":"B","because":"stated"}'), doctor_limit=1)
    assert recover_answer('My selected alternative is B because it approaches.', {'A':'away','B':'closer'}, doctor)['letter'] == 'B'
    assert doctor.doctor_calls == 1


def test_outer_trailing_comma_retains_tool_and_valid_entity():
    result = CallProcessor().process('{"tool":"track_entity","arguments":{"main":"runner"},}')
    assert result.call['tool'] == 'track_entity'
    assert result.call['arguments']['main'] == 'runner'


@pytest.mark.parametrize('text', ['The answer is A or B', 'Option A is wrong.', 'A or B.', 'The answer is A?'])
def test_doctor_cannot_promote_an_ambiguous_or_negated_letter(text):
    doctor = Scripted('{"letter":"A","because":"recovered"}')
    assert recover_answer(text, {'A':'left','B':'right'}, doctor) is None


def test_trailing_comma_repair_never_modifies_quoted_content():
    call, _ = normalize('{"tool":"ask_video","arguments":{"question":"Does ,} appear?",},}')
    assert call['arguments']['question'] == 'Does ,} appear?'


def test_conflicting_aliases_cannot_let_doctor_change_valid_main():
    doctor = Scripted('{"tool":"track_entity","arguments":{"main":"car","frame_of_reference":"world"}}')
    processor = CallProcessor(doctor)
    result = processor.process({'tool':'track_entity','arguments':{
        'main':'runner', 'frame':'subject', 'frame_of_reference':'world'}})
    assert result.call is None and 'conflicting' in result.message
    assert not doctor.calls


def test_json_answer_with_contradicting_tail_remains_unanswered():
    assert recover_answer('{"letter":"A"} Actually, the answer is B.', {'A':'left','B':'right'}) is None
