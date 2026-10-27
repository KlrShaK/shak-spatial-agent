"""V5 conversation storage with V6's protected evidence and keyframes."""
from d4rt_agent.agent_v5.context import Conversation as BaseConversation


class Conversation(BaseConversation):
    def fit(self, reserve=0):
        target = self.budget - reserve
        tokens = self.tokens()
        for index, turn in enumerate(self.turns):
            if tokens <= target:
                break
            if turn.kind == 'requested' and turn.images:
                before = tokens
                turn.images = []
                turn.trimmed = True
                tokens = self.tokens()
                self.trims.append(f'requested images {index}: {before} -> {tokens}')
        # Repeated calls can return byte-identical readings. Keep the complete
        # reading at its first occurrence and reference it without losing data.
        first_reports = {}
        for index, turn in enumerate(self.turns):
            if turn.kind != 'report':
                continue
            original = turn.summary if turn.trimmed and turn.summary is not None else turn.text
            first = first_reports.setdefault(original, index)
            if tokens > target and first != index and not turn.trimmed:
                before = tokens
                turn.summary = original
                turn.text = f'[Same report as turn {first}; the complete reading there is unchanged.]'
                turn.trimmed = True
                tokens = self.tokens()
                self.trims.append(f'duplicate report {index} references {first}: {before} -> {tokens}')
        # Keep the first schema menu and the current exchange. Older control
        # prompts, corrections and tool-call replies are not measured evidence.
        first_menu = next((turn for turn in self.turns if turn.kind == 'act' and turn.role == 'user'), None)
        for role, kinds in (('user', ('act', 'correction', 'budget')), ('assistant', ('act',))):
            for index, turn in enumerate(self.turns[:-2]):
                if tokens <= target:
                    break
                if turn.role != role or turn.kind not in kinds or turn is first_menu or turn.trimmed:
                    continue
                before = tokens
                turn.text = '[Earlier control turn omitted to fit the context budget.]'
                turn.trimmed = True
                tokens = self.tokens()
                self.trims.append(f'control turn {index} ({turn.kind}): {before} -> {tokens}')
        self.peak_tokens = max(self.peak_tokens, tokens)
        return tokens
