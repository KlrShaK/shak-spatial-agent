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
