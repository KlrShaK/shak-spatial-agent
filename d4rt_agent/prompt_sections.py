"""Section-addressable system prompts, so prompt changes are ablatable.

A prompt file marks its parts with HTML comments::

    <!-- section: coordinate_frame -->
    ## The coordinate frame
    ...
    <!-- endsection -->

The markers are invisible in rendered markdown and are stripped before the text
reaches Qwen, so a prompt with no ablations is byte-identical to the same file
with the markers removed.

The point is attribution.  The forced-grounding trial changed a rule *and*
rewrote every worked example in one commit, so when the result came back worse
there was no way to tell which half caused it.  Naming the parts makes
``--prompt-exclude examples`` a one-flag experiment.

Unknown section names raise rather than warn: an ablation that silently ran the
full prompt would be recorded as an ablation and quietly corrupt the comparison.
"""

from __future__ import annotations

from pathlib import Path
import re

_SECTION_RE = re.compile(
    r"<!--\s*section:\s*([A-Za-z0-9_]+)\s*-->\n(.*?)<!--\s*endsection\s*-->\n?",
    re.DOTALL,
)
_STRAY_MARKER_RE = re.compile(r"<!--\s*(?:section:\s*[A-Za-z0-9_]+|endsection)\s*-->")


def section_names(text: str) -> tuple[str, ...]:
    """Return the declared section names, in document order."""

    return tuple(match.group(1) for match in _SECTION_RE.finditer(text))


def _check_unbalanced(text: str) -> None:
    """Raise if a marker survives removal of every well-formed section."""

    stray = _STRAY_MARKER_RE.search(_SECTION_RE.sub("", text))
    if stray:
        raise ValueError(
            f"unbalanced prompt section marker: {stray.group(0)!r}. "
            "Every '<!-- section: name -->' needs a matching '<!-- endsection -->'."
        )


def render_prompt(
    text: str,
    *,
    exclude: tuple[str, ...] = (),
    override: dict[str, str] | None = None,
) -> str:
    """Resolve a sectioned prompt into the text Qwen receives.

    ``exclude`` drops sections entirely; ``override`` replaces a section's body
    with supplied text.  Both are keyed by section name.
    """

    _check_unbalanced(text)
    declared = section_names(text)
    duplicates = {name for name in declared if declared.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"duplicate prompt section name(s): {sorted(duplicates)}. "
            "Section names must be unique so an ablation addresses exactly one part."
        )

    override = dict(override or {})
    unknown = (set(exclude) | set(override)) - set(declared)
    if unknown:
        raise ValueError(
            f"unknown prompt section(s): {sorted(unknown)}. "
            f"Declared sections are: {list(declared)}"
        )
    both = set(exclude) & set(override)
    if both:
        raise ValueError(
            f"section(s) both excluded and overridden: {sorted(both)}. Choose one."
        )

    def _resolve(match: re.Match[str]) -> str:
        name, body = match.group(1), match.group(2)
        if name in exclude:
            return ""
        if name in override:
            return override[name].strip() + "\n"
        return body

    resolved = _SECTION_RE.sub(_resolve, text)
    resolved = _STRAY_MARKER_RE.sub("", resolved)
    return re.sub(r"\n{3,}", "\n\n", resolved).strip() + "\n"


def load_prompt(
    path: str | Path,
    *,
    exclude: tuple[str, ...] = (),
    override: dict[str, Path] | None = None,
) -> str:
    """Read a sectioned prompt file and resolve it against the ablation set."""

    text = Path(path).read_text(encoding="utf-8")
    bodies = {
        name: Path(source).read_text(encoding="utf-8")
        for name, source in (override or {}).items()
    }
    return render_prompt(text, exclude=exclude, override=bodies)


def parse_override_args(values: list[str] | None) -> dict[str, Path]:
    """Parse ``--prompt-override name=path`` arguments into a mapping."""

    parsed: dict[str, Path] = {}
    for raw in values or []:
        name, sep, source = raw.partition("=")
        if not sep or not name.strip() or not source.strip():
            raise ValueError(
                f"--prompt-override expects name=path, got {raw!r}"
            )
        path = Path(source.strip())
        if not path.is_file():
            raise ValueError(f"--prompt-override source does not exist: {path}")
        parsed[name.strip()] = path
    return parsed
