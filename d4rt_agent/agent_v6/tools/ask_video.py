"""Visual fact questions in a separate, stateless video conversation."""

from __future__ import annotations

from ..reports import Report
from ..workspace import WorkspaceError
from .captioner import PROMPTS, video_input


def ask_video(workspace, qwen, question):
    if not isinstance(question, str) or not question.strip():
        raise WorkspaceError("A visual question is required.")
    content, metadata = video_input(workspace, qwen)
    content.append({"type": "text", "text": question.strip()})
    messages = [{"role": "system", "content": (PROMPTS / "ask_video.md").read_text().strip()},
                {"role": "user", "content": content}]
    kwargs = {"max_new_tokens": 160}
    if metadata is not None:
        kwargs["video_metadata"] = metadata
    answer = qwen.generate(messages, **kwargs).strip()
    if not answer:
        return Report.unavailable("Video answer", "The video viewer returned no answer.")
    return Report("Video answer", headline=answer, raw={"question": question, "answer": answer})
