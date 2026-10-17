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
        self.peak_tokens = max(self.peak_tokens, tokens)
        return tokens
