"""Phased local-Qwen controller: entities, plain-text plan, calls, then answer."""
from __future__ import annotations

import json
from pathlib import Path
import re
import time

from .calls import CallProcessor, parse_object, recover_answer, valid_entity
from .context import Conversation
from .schemas import DESCRIPTIONS, schema, visible_tools

PROMPTS = Path(__file__).parent / 'prompts'


def prompt(name):
    return (PROMPTS / name).read_text()


def plan_items(text):
    """Read numbered lines for reminders, without imposing a plan schema."""
    return re.findall(r'^\s*\d+[.)]\s+(.+)', text, re.M)[:4]


class CountedQwen:
    def __init__(self, qwen):
        self.qwen = qwen
        self.calls = 0
        self.output_tokens = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        text = self.qwen.generate(messages, **kwargs)
        tokenizer = getattr(getattr(self.qwen, 'processor', None), 'tokenizer', None)
        self.output_tokens += len(tokenizer.encode(text)) if tokenizer else max(1, len(text) // 4)
        return text


class AgentV6:
    def __init__(self, *, qwen, max_steps=14, token_budget=32000, max_new_tokens=512, dispatcher=None, captioner=None):
        if not 1 <= max_steps <= 14:
            raise ValueError('max_steps must be between 1 and 14')
        self.qwen = CountedQwen(qwen)
        self.max_steps = max_steps
        self.token_budget = token_budget
        self.max_new_tokens = max_new_tokens
        self.dispatcher = dispatcher
        self.captioner = captioner

    def solve(self, task, *, workspace):
        from .toolset import Toolset
        from .tools.captioner import caption_clip

        started = time.monotonic()
        calls_before, tokens_before = self.qwen.calls, self.qwen.output_tokens
        workspace.begin_question()
        captioner = self.captioner or caption_clip
        caption = captioner(workspace, self.qwen)
        processor = getattr(self.qwen.qwen, 'processor', None)
        tokenizer = getattr(processor, 'tokenizer', None)
        count_text = (lambda text: len(tokenizer.encode(text))) if tokenizer is not None else None
        conversation = Conversation(prompt('system.md'), self.token_budget, count_text=count_text)
        images = [(workspace.mark_shown(i), workspace.clip.pil(i, long_side=448))
                  for i in workspace.clip.keyframes(8)]
        question = task['question'] + '\nOptions: ' + json.dumps(task['options'])
        conversation.add('user', 'Caption: ' + caption + '\n' + question, images=images, kind='opening')
        trace, reports = [], []
        doctor = CallProcessor(self.qwen)

        def generate(phase, instruction):
            conversation.add('user', instruction, kind=phase)
            used = conversation.fit(reserve=self.max_new_tokens)
            if used + self.max_new_tokens > self.token_budget:
                raise ValueError('Protected evidence exceeds the context budget; no evidence was discarded.')
            response = self.qwen.generate(conversation.messages(), max_new_tokens=self.max_new_tokens)
            conversation.add('assistant', response, kind=phase)
            trace.append({'phase': phase, 'response': response})
            return response

        entities_text = generate('entities', prompt('entities.md'))
        for attempt in range(2):
            try:
                entities = parse_object(entities_text)['entities']
                if not isinstance(entities, list) or not all(valid_entity(n) for n in entities):
                    raise ValueError('Use a list of short descriptive entity names (at most five words each).')
                for name in entities:
                    workspace.register(name)
                break
            except (ValueError, KeyError, TypeError):
                if attempt:
                    entities = []
                    trace.append({'phase': 'entities', 'unavailable': 'entity response remained malformed'})
                    break
                entities_text = generate('entities', 'Return only {"entities": ["short descriptive name"]}; camera is implicit.')
        menu = '\n'.join(f'{name}: {DESCRIPTIONS[name]}' for name in visible_tools())
        plan = generate('plan', prompt('plan.md') + '\nAvailable measurements:\n' + menu)
        items = plan_items(plan)
        toolset = self.dispatcher or Toolset(workspace, self.qwen, task['question'], task['options'], caption)
        hints = json.loads(prompt('hints.json'))
        seen_tools, used, active_item = set(), 0, 0
        executed_items = set()
        answer = None
        plan_revision = 0
        schemas_sent = False
        # Invalid calls and revisions do not consume measurement budget; cap turns to ensure termination.
        for turn in range(self.max_steps + 12):
            if used >= self.max_steps:
                break
            pending = workspace.pending_candidates is not None
            names = visible_tools(pending)
            item = items[min(active_item, len(items) - 1)] if items else 'Continue or revise the plan.'
            category = re.search(r'\[([^]]+)\]', item)
            hint = hints.get(category[1], '') if category else ''
            reminder = f'Plan item {min(active_item + 1, max(1, len(items)))} of {len(items)} next: {item} · calls used {used}/{self.max_steps}. {hint}'
            if used >= self.max_steps - 3:
                reminder += f' WARNING: {self.max_steps - used} tool calls remain; answer when evidence is sufficient.'
            if not schemas_sent:
                menu_text = json.dumps([schema(name) for name in names])
                schemas_sent = True
            else:
                menu_text = '\n'.join(f'{name}: {DESCRIPTIONS[name]}' for name in names)
                if pending:
                    menu_text += '\n' + json.dumps(schema('choose_segmentation_candidate'))
            response = generate('act', reminder + '\n' + menu_text +
                                '\nWhen a planned quantity is established, include completed_plan_items: [item numbers] in your next JSON. Auxiliary calls alone do not complete a quantity. Continue with one {"tool": "<name>", "arguments": {...}}; revise the plan, or return {"answer": true}.')
            if response.lstrip().lower().startswith('revised plan:'):
                plan = response
                items = plan_items(plan)
                active_item = 0
                executed_items.clear()
                plan_revision += 1
                trace.append({'phase': 'plan_revision', 'revision': plan_revision, 'plan': plan})
                continue
            try:
                action = parse_object(response)
            except ValueError:
                action = {}
            completed = action.get('completed_plan_items', [])
            if isinstance(completed, list):
                executed_items.update(i for i in completed if isinstance(i, int) and not isinstance(i, bool) and 1 <= i <= len(items))
                active_item = next((i for i in range(len(items)) if i + 1 not in executed_items), max(0, len(items) - 1))
            if action.get('answer') is True or action.get('action') in ('answer', 'final_answer') or 'letter' in action:
                # A phase change always gives the model the answer precedence and format instructions.
                break
            processed = doctor.process(response, names=workspace.known_names, pending=pending, plan=plan,
                                       last_turns=[t.text for t in conversation.turns[-2:]])
            trace.append({'phase': 'call', 'steps': processed.trace, 'message': processed.message})
            if processed.message:
                conversation.add('user', processed.message, kind='correction')
            if processed.call is None:
                continue
            used += 1
            tool = processed.call['tool']
            workspace.begin_tool_call(tool)
            report = toolset.call(tool, **processed.call['arguments'])
            rendered = report.render(full=tool not in seen_tools)
            seen_tools.add(tool)
            conversation.add('user', rendered, kind='report')
            if report.images:
                conversation.add('user', 'Requested views from ' + tool, images=report.images, kind='requested')
            reports.append({'tool': tool, 'title': report.title, 'headline': report.headline,
                            'status': report.status, 'reason': report.reason, 'raw': report.raw,
                            'rendered': rendered})
            trace.append({'phase': 'report', 'tool': tool, 'status': report.status,
                          'call_number': used, 'plan_item': active_item + 1, 'plan_revision': plan_revision})
            if used >= self.max_steps - 2:
                conversation.add('user', f'WARNING: call {used}/{self.max_steps} used; {self.max_steps - used} remain.', kind='budget')
        final = generate('answer', prompt('answer.md') + '\n' + question)
        answer = recover_answer(final, task['options'], doctor if doctor.doctor_calls < 5 else None)
        letter = answer['letter'] if answer else None
        because = answer['because'] if answer else ''
        return {'question_id': task['id'], 'status': 'ok' if answer else 'unanswered', 'answered': bool(answer),
                'letter': letter, 'because': because,
                'answer': f'{letter}: {task["options"][letter]} — {because}' if answer else None,
                'caption': caption, 'entities': entities, 'plan': plan, 'reports': reports, 'trace': trace,
                'tool_calls': used, 'model_calls': self.qwen.calls - calls_before,
                'output_tokens': self.qwen.output_tokens - tokens_before,
                'peak_context_tokens': conversation.peak_tokens, 'trims': conversation.trims,
                'process': dict(doctor.counts, doctor_calls=doctor.doctor_calls,
                                planned_items=len(items), executed_items=len(executed_items)),
                'wall_seconds': time.monotonic() - started}
