"""The conversation, kept under a token budget so an 8B model never reasons over a bloated context.

One conversation per question (as in v4), but every turn is a ``Turn`` record that knows what it
is and how to say itself in one line. Before each generation the host estimates the context size;
above the budget it replaces the oldest expendable turns with their summaries, in this order:
images from ``show`` -> full tool tables -> rejected-action exchanges. The opening turn (question
and keyframes) is never trimmed, and the answer turn re-renders every tool table from the host's
own record, so nothing measured is lost by trimming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

TRIM_ORDER = ("show", "tool", "reject")
CHARS_PER_TOKEN = 3.6
IMAGE_PATCH_PX = 32                 # Qwen3-VL: 16 px patches merged 2x2


@dataclass
class Turn:
    role: str                       # "system" | "user" | "assistant"
    text: str
    images: list[Any] = field(default_factory=list)
    kind: str = "other"             # "opening" | "tool" | "show" | "reject" | "step" | "answer" | "other"
    summary: str | None = None
    trimmed: bool = False

    def content(self) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for image in self.images:
            if isinstance(image, tuple):                       # (label, image)
                parts.append({"type": "text", "text": image[0]})
                parts.append({"type": "image", "image": image[1]})
            else:
                parts.append({"type": "image", "image": image})
        parts.append({"type": "text", "text": self.text})
        return parts


def image_tokens(image: Any) -> int:
    width, height = getattr(image, "size", (448, 448))
    return max(1, round(width / IMAGE_PATCH_PX)) * max(1, round(height / IMAGE_PATCH_PX)) + 2


class Conversation:
    def __init__(self, system: str, budget: int, count_text: Callable[[str], int] | None = None) -> None:
        self.turns: list[Turn] = [Turn("system", system, kind="system")]
        self.budget = int(budget)
        self._count = count_text or (lambda text: int(len(text) / CHARS_PER_TOKEN) + 1)
        self.trims: list[str] = []
        self.peak_tokens = 0

    def add(self, role: str, text: str, *, images: list[Any] | None = None, kind: str = "other",
            summary: str | None = None) -> Turn:
        turn = Turn(role, text, list(images or []), kind, summary)
        self.turns.append(turn)
        return turn

    def messages(self) -> list[dict[str, Any]]:
        return [{"role": t.role, "content": t.content()} for t in self.turns]

    def tokens(self) -> int:
        total = 0
        for turn in self.turns:
            total += self._count(turn.text) + 4
            for image in turn.images:
                total += image_tokens(image[1] if isinstance(image, tuple) else image)
        return total

    def fit(self) -> int:
        """Trim until the estimate is under the budget (or nothing is left to trim); the estimate."""

        tokens = self.tokens()
        for kind in TRIM_ORDER:
            for index, turn in enumerate(self.turns[:-1]):          # never the newest turn
                if tokens <= self.budget:
                    break
                if turn.trimmed or turn.kind != kind:
                    continue
                before = tokens
                turn.images = []
                if turn.summary:
                    turn.text = f"[earlier result, shortened to save space] {turn.summary}"
                turn.trimmed = True
                tokens = self.tokens()
                self.trims.append(f"turn {index} ({kind}): {before} -> {tokens} tokens")
        self.peak_tokens = max(self.peak_tokens, tokens)
        return tokens
