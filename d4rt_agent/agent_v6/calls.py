"""Deterministic call repair followed by a bounded, value-preserving local doctor."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
import re

from d4rt_agent.simple_v2 import extract_first_action
from .schemas import DEFAULTS, FIELDS, REQUIRED, schema, visible_tools


def parse_object(raw):
    if isinstance(raw, dict):
        return dict(raw)
    # Inspect complete outer objects only. An invalid envelope must never turn
    # its nested arguments into a new action and lose already valid values.
    depth, start, quote, escaped = 0, None, None, False
    for index, char in enumerate(raw):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if depth and char in ('"', "'"):
            quote = char
        elif char == '{':
            if depth == 0:
                start = index
            depth += 1
        elif char == '}' and depth:
            depth -= 1
            if depth == 0:
                candidate = raw[start:index + 1]
                # literal_eval accepts single quotes/trailing commas without
                # executing code; JSON is preferred for true/false/null.
                for loader in (json.loads, ast.literal_eval):
                    try:
                        value = loader(candidate)
                        if isinstance(value, dict):
                            return value
                    except (ValueError, SyntaxError):
                        pass
                # Repair trailing commas outside strings, retaining all values.
                clean, inside, escape = [], False, False
                for pos, token in enumerate(candidate):
                    if inside:
                        clean.append(token)
                        if escape:
                            escape = False
                        elif token == "\\":
                            escape = True
                        elif token == '"':
                            inside = False
                    elif token == '"':
                        inside = True
                        clean.append(token)
                    elif token != ',' or candidate[pos + 1:].lstrip()[:1] not in (']', '}'):
                        clean.append(token)
                try:
                    repaired = ''.join(clean)
                    try:
                        outer = json.loads(repaired)
                        # Reuse the legacy extractor only after the complete
                        # envelope has been decoded, never for nested salvage.
                        return extract_first_action(repaired).value if 'action' in outer or 'name' in outer else outer
                    except ValueError:
                        value = json.loads(repaired)
                        if isinstance(value, dict):
                            return value
                except ValueError:
                    pass
                raise ValueError('Malformed outer JSON call; preserve its tool and arguments when correcting it.')
    raise ValueError('Return one JSON object with tool and arguments.')


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return row[-1]


def close_match(value, choices):
    if value in choices:
        return value
    distances = [(edit_distance(value, choice), choice) for choice in choices]
    if not distances:
        return value
    best = min(d for d, _ in distances)
    matches = [choice for d, choice in distances if d == best]
    return matches[0] if len(matches) == 1 and best <= max(1, min(2, len(value) // 4)) else value


def entity_name(value, names):
    if not isinstance(value, str):
        return value
    clean = re.sub(r'^(the|a|an)\s+', '', value.strip().lower())
    if clean in ('cam', 'camera', 'observer'):
        return 'camera'
    known = {re.sub(r'^(the|a|an)\s+', '', n.lower()): n for n in names}
    return known.get(clean, value.strip())


def normalize(raw, names=()):
    value = parse_object(raw)
    tool = value.get('tool', value.get('name', value.get('action', '')))
    if not isinstance(tool, str):
        raise ValueError('tool must be a name in the tool menu')
    tool = close_match(tool, FIELDS)
    arguments = value.get('arguments', value.get('args',
                          {k: v for k, v in value.items() if k not in ('tool', 'name', 'action', 'thought', 'completed_plan_items')}))
    if not isinstance(arguments, dict):
        raise ValueError('arguments must be a JSON object')
    fixes, result = [], {}
    aliases = {'frame': 'frame_of_reference', 'reference_frame': 'frame_of_reference', 'color': 'colour'}
    for key, item in arguments.items():
        new_key = aliases.get(key, close_match(key, FIELDS.get(tool, {})))
        if new_key in result:
            raise ValueError(f'duplicate argument after normalization: {new_key}')
        new_item = item
        if new_key == 'frame_of_reference' and isinstance(item, str):
            new_item = {'subj': 'subject', 'cam': 'camera', 'scene': 'world'}.get(item, item)
            allowed = FIELDS.get(tool, {}).get(new_key)
            if isinstance(allowed, tuple):
                new_item = close_match(new_item, allowed)
            else:
                new_item = entity_name(new_item, names)
        elif new_key in ('main', 'category'):
            new_item = entity_name(item, names)
        elif new_key in ('others', 'entities') and isinstance(item, list):
            new_item = [entity_name(n, names) for n in item]
        result[new_key] = new_item
        if key != new_key or item != new_item:
            fixes.append(f'{key} → {new_key}={new_item}')
    for key, item in DEFAULTS.get(tool, {}).items():
        if key not in result:
            result[key] = item.copy() if isinstance(item, list) else item
            fixes.append(f'default {key}={item}')
    if value.get('tool') != tool or 'arguments' not in value:
        fixes.append('call envelope → tool/arguments')
    return {'tool': tool, 'arguments': result}, fixes


def valid_entity(value):
    return (isinstance(value, str) and 0 < len(value.split()) <= 5 and
            bool(re.fullmatch(r"[\w\s'’\-]+", value)) and
            not re.fullmatch(r'(?:[ec]\d+|\d+|(?:inspection|visual|view)[_\-]?\d+)', value, re.I))


def field_valid(kind, value):
    if isinstance(kind, tuple):
        return isinstance(value, str) and value in kind
    if kind == 'entity':
        return valid_entity(value)
    if kind == 'orientation_reference':
        return value in ('camera', 'own') if isinstance(value, str) and value in ('camera', 'own') else valid_entity(value)
    if kind == 'entities':
        return isinstance(value, list) and all(valid_entity(v) for v in value)
    if kind == 'text':
        return isinstance(value, str) and bool(value.strip())
    if kind.startswith('times') or kind == 'range':
        limit = int(kind[5:]) if kind.startswith('times') else 2
        return (isinstance(value, list) and 1 <= len(value) <= limit and
                (kind != 'range' or len(value) == 2) and
                all((isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v < float('inf'))
                    or (isinstance(v, str) and re.fullmatch(r'shown frame [A-Z]+', v, re.I)) for v in value))
    return False


def validate(call, pending=False):
    tool, args = call['tool'], call['arguments']
    if tool not in visible_tools(pending):
        return ['Tool is unavailable in this phase; use the visible menu.']
    errors = [f'Missing {key}.' for key in REQUIRED.get(tool, ()) if key not in args]
    for key, value in args.items():
        if key not in FIELDS[tool]:
            errors.append(f'Unknown {key}; use seconds or shown frame labels, never generator frames or inspection handles.')
        elif not field_valid(FIELDS[tool][key], value):
            errors.append(f'Invalid {key}={value!r}; expected {FIELDS[tool][key]}. Use names in words.')
    if tool == 'get_frames' and 'times' in args and 'range' in args:
        errors.append('Choose times or range, not both.')
    if tool == 'get_segmentation_mask' and args.get('entities') == []:
        errors.append('entities must be nonempty; use get_frames for plain frames.')
    if tool == 'track_entity' and args.get('main') == 'camera':
        errors.append('Use track_camera for the camera itself.')
    return errors


@dataclass
class CallResult:
    call: dict | None
    message: str = ''
    trace: list = field(default_factory=list)


class CallProcessor:
    def __init__(self, qwen=None, doctor_limit=5):
        self.qwen = qwen
        self.doctor_limit = doctor_limit
        self.doctor_calls = 0
        self.counts = dict(normalizer_fixes=0, doctor_fixes=0, sent_back=0)

    def generate(self, messages):
        if self.qwen is None or self.doctor_calls >= self.doctor_limit:
            raise ValueError('Doctor budget exhausted')
        self.doctor_calls += 1
        return self.qwen.generate(messages)

    def process(self, raw, *, names=(), pending=False, plan='', last_turns=()):
        trace = [{'stage': 'raw', 'value': raw}]
        try:
            call, fixes = normalize(raw, names)
            errors = validate(call, pending)
        except (ValueError, TypeError) as error:
            call, fixes, errors = None, [], [str(error)]
            if 'duplicate argument after normalization' in str(error):
                self.counts['sent_back'] += 1
                trace.append({'stage': 'conflicting_arguments', 'error': str(error)})
                return CallResult(None, 'To the agent: conflicting argument aliases; keep one frame_of_reference and preserve your other values.', trace)
            # Preserve canonical, independently valid values if other parts of
            # a malformed envelope could not be normalized.
            try:
                original = parse_object(raw)
                tool = original.get('tool', original.get('name', original.get('action')))
                args = original.get('arguments', original.get('args', {}))
                if tool in FIELDS and isinstance(args, dict):
                    call = {'tool': tool, 'arguments': {k: v for k, v in args.items()
                            if k in FIELDS[tool] and field_valid(FIELDS[tool][k], v)}}
            except (ValueError, TypeError):
                pass
        trace.append({'stage': 'normalized', 'value': call, 'fixes': fixes, 'errors': errors})
        if fixes:
            self.counts['normalizer_fixes'] += 1
        if not errors:
            return CallResult(call, 'corrected: ' + '; '.join(fixes) if fixes else '', trace)
        if self.qwen and self.doctor_calls < self.doctor_limit:
            from pathlib import Path
            instruction = (Path(__file__).parent / 'prompts/doctor.md').read_text()
            selected = schema(call['tool']) if call and call['tool'] in FIELDS else list(visible_tools(pending))
            context = dict(raw=raw, errors=errors, plan=plan, last_turns=list(last_turns)[-2:], schema=selected)
            reply = self.generate([{'role': 'system', 'content': instruction},
                                        {'role': 'user', 'content': json.dumps(context, default=str)}])
            try:
                repaired, _ = normalize(reply, names)
                failure = validate(repaired, pending)
                if call and call['tool'] in FIELDS:
                    if repaired['tool'] != call['tool']:
                        failure.append('doctor changed a valid tool')
                    for key, value in call['arguments'].items():
                        if key in FIELDS[call['tool']] and field_valid(FIELDS[call['tool']][key], value):
                            if repaired['arguments'].get(key) != value:
                                failure.append(f'doctor changed valid {key}')
                if not failure:
                    self.counts['doctor_fixes'] += 1
                    trace.append({'stage': 'doctor', 'value': repaired})
                    return CallResult(repaired, 'corrected by doctor: ' + json.dumps(repaired), trace)
                errors += failure
            except (ValueError, TypeError) as error:
                errors.append(str(error))
            trace.append({'stage': 'doctor_rejected', 'value': reply, 'errors': errors})
        self.counts['sent_back'] += 1
        return CallResult(None, 'To the agent: ' + ' '.join(errors), trace)


def _declared_letters(raw, options, extended=False):
    text = str(raw)
    noun = r'(?:answer|choice|letter' + (r'|selected alternative' if extended else '') + r')'
    pattern = (r'(?im)(?:^\s*|\b' + noun + r'\s*(?:is|was|:|=)\s*)'
               r'["\']?([A-Z])["\']?(?=\b)')
    letters = set()
    ambiguous = False
    for match in re.finditer(pattern, text):
        letter = match[1].upper()
        tail = text[match.end():]
        if letter not in options:
            continue
        if re.match(r'\s*(?:(?:or|and|/)\s*(?:option\s*)?[A-Z]\b|\?|(?:is\s+)?(?:wrong|incorrect|not)\b)', tail, re.I):
            ambiguous = True
        else:
            letters.add(letter)
    return None if ambiguous else letters


def recover_answer(raw, options, doctor=None):
    """Recover an explicit choice only; a doctor cannot supply a missing decision."""
    try:
        value = parse_object(raw)
    except (ValueError, TypeError):
        value = {}
    letter = value.get('letter')
    if isinstance(letter, str) and letter.strip().upper() in options:
        declarations = _declared_letters(raw, options, extended=True)
        if declarations is None or (declarations and declarations != {letter.strip().upper()}):
            return None
        return {'letter': letter.strip().upper(), 'because': str(value.get('because', ''))}
    letters = _declared_letters(raw, options)
    if letters and len(letters) == 1:
        return {'letter': letters.pop(), 'because': str(raw)}
    if doctor and letters == set():
        reply = doctor.generate([{'role': 'system', 'content': 'Recover an explicitly stated answer only. Return its letter and because as JSON, or null. Never choose or infer an answer.'},
                                 {'role': 'user', 'content': str(raw)}])
        try:
            recovered = parse_object(reply)
        except (ValueError, TypeError):
            return None
        candidate = recovered.get('letter')
        evidence = _declared_letters(raw, options, extended=True)
        if candidate in options and evidence == {candidate}:
            return {'letter': candidate, 'because': str(recovered.get('because', ''))}
    return None
