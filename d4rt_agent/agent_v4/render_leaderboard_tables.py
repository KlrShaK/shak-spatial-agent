"""Render the two "Against the published leaderboard" tables in
`d4rt_agent/results/official_export/PER_CATEGORY_VS_PUBLISHED.md` as PNGs.

Reads the markdown directly rather than hardcoding the numbers a second time,
so the images never drift from the doc -- regenerate after any edit to those
two tables (`## Against the published leaderboard` / `### Sample-wise` and its
sibling `### Group-wise (>=3/4 augmentations correct)`).

Styling:
  * a row already bolded in the source (our own rows -- Control, Ours v4) keeps
    bold text in the image.
  * per column, the 1st/2nd/3rd-highest cells get a highlight, darkest for 1st,
    fading for 2nd and 3rd (`--sequential hue` steps 450/300/150 from the
    project's dataviz palette: d4rt_agent/README.md has no palette of its own,
    so this borrows the validated default rather than inventing one).
    Ties share a rank's color -- e.g. two rows both printing "50.00" in one
    column both get the rank-1 highlight.
  * "-" cells (a model with no reported score in that category, e.g.
    SpatialTrackerV2's Orientation column) are left out of the ranking and
    rendered as plain "-".
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# So a bolded venue name in the footer (via mathtext \mathbf, see
# _bold_md_to_mathtext) renders in the same sans-serif family as the rest of
# the figure instead of mathtext's serif-ish default (Computer Modern).
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"

DOC = Path("d4rt_agent/results/official_export/PER_CATEGORY_VS_PUBLISHED.md")
OUT_DIR = Path("d4rt_agent/results/official_export")

# Sequential blue ramp, steps 450/300/150 -- see the dataviz skill's
# references/palette.md. 1st gets the darkest step (white text), 2nd/3rd get
# progressively lighter steps (dark text).
RANK_COLORS = ["#2a78d6", "#6da7ec", "#b7d3f6"]
RANK_TEXT = ["#ffffff", "#0b0b0b", "#0b0b0b"]

MISSING = {"—", "-", "–", "--", ""}


class Table(NamedTuple):
    heading: str
    columns: list[str]              # header cells, "Model" + one per category
    rows: list[str]                 # cleaned row labels
    bold_row: list[bool]            # row was bold in the source (our rows)
    cells: list[list[str]]          # cleaned display text, rows x (columns-1)
    values: list[list[float | None]]  # parsed numeric value, same shape


def _strip_md(cell: str) -> tuple[str, bool]:
    """Return (display text, was this cell bold in the source)."""
    cell = cell.strip()
    bold = "**" in cell
    cell = re.sub(r"\*\*(.+?)\*\*", r"\1", cell)
    cell = re.sub(r"\*(.+?)\*", r"\1", cell)
    return cell.strip(), bold


def _parse_value(text: str) -> float | None:
    if text.strip() in MISSING:
        return None
    try:
        return float(text.replace("%", "").strip())
    except ValueError:
        return None


def extract_table(md_lines: list[str], heading: str) -> Table:
    """Pull the first markdown table that follows an exact heading line."""

    start = next(i for i, line in enumerate(md_lines) if line.strip() == heading)
    table_lines = []
    for line in md_lines[start + 1:]:
        if line.startswith("|"):
            table_lines.append(line)
        elif table_lines:
            break
    if len(table_lines) < 3:
        raise ValueError(f"no table found under heading {heading!r}")

    def split_row(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header_cells = split_row(table_lines[0])
    columns = [_strip_md(c)[0] for c in header_cells]
    # table_lines[1] is the "| --- | ---: | ..." separator -- skip it.

    rows, bold_row, cells, values = [], [], [], []
    for line in table_lines[2:]:
        raw = split_row(line)
        label, label_bold = _strip_md(raw[0])
        rows.append(label)
        bold_row.append(label_bold)
        row_cells, row_values = [], []
        for c in raw[1:]:
            text, _ = _strip_md(c)
            row_cells.append(text)
            row_values.append(_parse_value(text))
        cells.append(row_cells)
        values.append(row_values)

    return Table(heading, columns, rows, bold_row, cells, values)


def extract_footer_paragraphs(md_lines: list[str], after_heading: str,
                               stop_heading: str) -> list[str]:
    """Paragraphs between the table under `after_heading` and `stop_heading`.

    Used for the footnotes under the sample-wise table (the PolyV/DynTrace
    provenance notes) -- skips the table rows themselves (lines starting with
    "|"), joins each blank-line-delimited paragraph into one string, and
    strips markdown emphasis so it prints cleanly in the image.
    """

    start = next(i for i, line in enumerate(md_lines) if line.strip() == after_heading)
    stop = next(i for i, line in enumerate(md_lines) if line.strip() == stop_heading)

    paragraphs, current = [], []
    for line in md_lines[start + 1:stop]:
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if stripped.startswith("|"):
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))

    return paragraphs  # markdown emphasis left in -- render() bolds it


def _bold_md_to_mathtext(text: str) -> str:
    """Turn markdown **bold** spans (the publication venue in a footnote)
    into matplotlib mathtext bold, leaving the rest as plain text.

    Spaces inside the span become "\\ " so the whole span survives
    textwrap.wrap() as one atomic word -- otherwise a line break could land
    inside "$\\mathbf{...}$" and break the math parsing.
    """

    def repl(m: re.Match[str]) -> str:
        inner = m.group(1).replace(" ", r"\ ")
        return rf"$\mathbf{{{inner}}}$"

    return re.sub(r"\*\*(.+?)\*\*", repl, text)


def _wrap_to_pixel_width(paragraphs: list[str], max_width_px: float,
                          fontsize: int, fig, renderer) -> list[str]:
    """Word-wrap each paragraph to a *measured* pixel width, not a guessed
    chars-per-inch ratio -- guarantees no line renders wider than
    `max_width_px` regardless of font metrics or mathtext spans.
    """

    scratch = fig.text(0, 0, "", fontsize=fontsize)
    lines: list[str] = []
    for i, para in enumerate(paragraphs):
        if i:
            lines.append("")
        words = para.split(" ")
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            scratch.set_text(candidate)
            width_px = scratch.get_window_extent(renderer).width
            if width_px <= max_width_px or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    scratch.remove()
    return lines


def rank_colors_for_column(values: list[float | None]) -> dict[int, str]:
    """row-index -> hex color for the top-3 *distinct* values in this column."""

    present = sorted({v for v in values if v is not None}, reverse=True)
    top3 = present[:3]
    value_to_color = dict(zip(top3, RANK_COLORS))
    return {i: value_to_color[v] for i, v in enumerate(values) if v in value_to_color}


def rank_text_for_column(values: list[float | None]) -> dict[int, str]:
    present = sorted({v for v in values if v is not None}, reverse=True)
    top3 = present[:3]
    value_to_text = dict(zip(top3, RANK_TEXT))
    return {i: value_to_text[v] for i, v in enumerate(values) if v in value_to_text}


def render(table: Table, out_path: Path, title: str, highlight: bool = True,
           footer_paragraphs: list[str] | None = None) -> None:
    n_rows, n_cols = len(table.rows), len(table.columns)
    # Sized from content, not a fixed guess: the label column carries long
    # annotated names ("... (full census, n=7,076)"), and a fixed-fraction
    # width either clips them or starves the numeric columns.
    label_chars = max(len(r) for r in table.rows + [table.columns[0]])
    data_chars = max(len(c) for c in table.columns[1:])
    fig_w = 0.11 * label_chars + 0.10 * data_chars * (n_cols - 1) + 0.6
    table_h = 0.9 + 0.42 * (n_rows + 1)

    # Footer text lives on the *same* axes as the table, placed just below
    # the table's actual rendered bbox (measured after a draw pass, below) --
    # not in a second axes sized by a guessed table height. A table's natural
    # height is rarely equal to any such guess, and a second axes leaves that
    # difference as visible dead space between the two.
    # This first wrap is only to reserve enough figure height -- deliberately
    # narrower than the table (0.75x) so it over-, not under-, counts lines;
    # the real wrap (against the table's *measured* width, guaranteeing no
    # overflow) happens after the table is drawn, below.
    FOOTER_FONTSIZE = 10
    footer_lines: list[str] = []
    if footer_paragraphs:
        mathtext_paragraphs = [_bold_md_to_mathtext(p) for p in footer_paragraphs]
        chars_per_inch = 155 / FOOTER_FONTSIZE * 0.75
        wrap_width = max(40, int(fig_w * chars_per_inch))
        for i, para in enumerate(mathtext_paragraphs):
            if i:
                footer_lines.append("")
            footer_lines.extend(textwrap.wrap(para, width=wrap_width))
        footer_h = 0.1 + FOOTER_FONTSIZE / 72 * 1.5 * len(footer_lines)
    else:
        footer_h = 0.0

    fig_h = table_h + footer_h
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14, loc="left")

    header = table.columns
    body = [[r] + c for r, c in zip(table.rows, table.cells)]
    mpl_table = ax.table(cellText=body, colLabels=header, cellLoc="center",
                          loc="upper left")
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(9)
    mpl_table.auto_set_column_width(col=list(range(n_cols)))
    mpl_table.scale(1, 1.6)

    # Recessive grid (thin, light gray) instead of matplotlib's default heavy
    # black borders -- fill colors read poorly against a black grid.
    for cell in mpl_table.get_celld().values():
        cell.set_edgecolor("#c3c2b7")
        cell.set_linewidth(0.6)

    # Header styling.
    for col in range(n_cols):
        cell = mpl_table[0, col]
        cell.set_text_props(fontweight="bold", color="#0b0b0b")
        cell.set_facecolor("#e1e0d9")

    # Per-column rank highlight, computed over each category column (skip the
    # label column, index 0). Skipped for small excerpt tables where "top 3
    # of 1 or 2 rows" isn't a ranking, just zebra striping for readability.
    for col in range(1, n_cols):
        col_values = [row_vals[col - 1] for row_vals in table.values]
        colors = rank_colors_for_column(col_values) if highlight else {}
        text_colors = rank_text_for_column(col_values) if highlight else {}
        for row in range(n_rows):
            cell = mpl_table[row + 1, col]
            if row in colors:
                cell.set_facecolor(colors[row])
                cell.get_text().set_color(text_colors[row])
            elif row % 2 == 1:
                cell.set_facecolor("#f9f9f7")  # faint zebra stripe

    # Row label column + bold for our own rows, applied last so it overrides
    # nothing (label column never carries a rank highlight).
    for row in range(n_rows):
        label_cell = mpl_table[row + 1, 0]
        label_cell.set_text_props(ha="left")
        label_cell.get_text().set_x(0.02)
        if table.bold_row[row]:
            for col in range(n_cols):
                mpl_table[row + 1, col].set_text_props(fontweight="bold")

    if footer_lines:
        # Measure the table's *actual* rendered bbox (a draw pass is needed
        # first -- auto_set_column_width/scale don't finalize cell geometry
        # until then) and place the footer directly below it, rather than
        # below a guessed table height. tight_layout is skipped on this path:
        # it can reposition the axes, which would invalidate the just-measured
        # bbox; bbox_inches="tight" on save does the real trimming anyway.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = mpl_table.get_window_extent(renderer)
        (_, table_bottom), (_, _) = ax.transAxes.inverted().transform(
            [(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])

        # Re-wrap against the table's *measured* pixel width (a 2% inset so
        # text never touches the table's own edges) -- replaces the rough
        # upfront wrap, which only had to be narrow enough, not exact.
        footer_lines = _wrap_to_pixel_width(
            mathtext_paragraphs, bbox.width * 0.98, FOOTER_FONTSIZE, fig, renderer)

        gap = 0.4 / table_h  # ~0.4in visual gap, in this axes' own fraction
        ax.text(0, table_bottom - gap, "\n".join(footer_lines),
                fontsize=FOOTER_FONTSIZE, color="#52514e", va="top", ha="left",
                linespacing=1.5, transform=ax.transAxes)
        fig.savefig(out_path, bbox_inches="tight")
    else:
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=DOC)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    md_lines = args.doc.read_text().splitlines()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sample_wise = extract_table(md_lines, "### Sample-wise")
    footnotes = extract_footer_paragraphs(
        md_lines, "### Sample-wise", "### Their own detailed breakdowns")
    render(sample_wise, args.out_dir / "leaderboard_sample_wise.png",
           "DSI-Bench leaderboard — sample-wise (each augmentation independent)",
           footer_paragraphs=footnotes)

    # Small excerpt tables from two external papers, subset to their own
    # method's row(s) -- not a leaderboard, so no rank highlight.
    polyv = extract_table(md_lines, "#### PolyV")
    render(polyv, args.out_dir / "leaderboard_polyv_subset.png",
           "PolyV — DSI-Bench by video source (their own row only)",
           highlight=False)

    dyntrace = extract_table(md_lines, "#### DynTrace")
    render(dyntrace, args.out_dir / "leaderboard_dyntrace_subset.png",
           "DynTrace — DSI-Bench by category (their own rows only)",
           highlight=False)

    group_wise = extract_table(md_lines, "### Group-wise (≥3/4 augmentations correct)")
    render(group_wise, args.out_dir / "leaderboard_group_wise.png",
           "DSI-Bench leaderboard — group-wise (≥3/4 augmentations correct)")


if __name__ == "__main__":
    main()
