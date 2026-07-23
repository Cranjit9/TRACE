#!/usr/bin/env python3
"""PyMuPDF composer for main figures 2-4 and supp figure S6."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator
import fitz

STYLES: dict[str, dict] = {
    "figure_2": {
        "page_width": 1152.0,
        "margin": (18.0, 18.0),
        "gap": (0.0, 25.0),
        "letter_fontsize": 20.0,
        "letter_color": (0.15, 0.15, 0.15),
        "letter_offset_y": 4.0,
        "letters": "ABC",
    },
    "figure_3": {
        "page_width": 1152.0,
        "margin": (18.0, 18.0),
        "gap": (20.0, 30.0),
        "letter_fontsize": 18.0,
        "letter_color": (0.15, 0.15, 0.15),
        "letter_offset_y": 4.0,
        "letters": "ABCDEF",
    },
    "figure_4": {
        "page_width": 960.0,
        "margin": (28.0, 28.0),
        "gap": (18.0, 12.0),
        "bc_gap": 6.0,
        "fixed_a_width": 500.0,
        "letter_fontsize": 14.0,
        "letter_color": (0.0, 0.0, 0.0),
        "letter_offset_y": 4.0,
        "letter_offsets_below_top": {"F": 12.0, "G": 12.0},
        "letters": "ABCDEFG",
    },
    "figure_S6": {
        "page_width": 960.0,
        "margin": (28.0, 28.0),
        "gap": (18.0, 12.0),
        "fixed_a_width": 500.0,
        "letter_fontsize": 14.0,
        "letter_color": (0.0, 0.0, 0.0),
        "letter_offset_y": 4.0,
        "letter_offsets_below_top": {"B": 12.0, "C": 12.0},
        "letters": "ABC",
    },
}

@contextmanager
def _open_sources(panel_paths: dict, letters: str) -> Iterator[dict]:
    src: dict = {}
    try:
        for L in letters:
            path = Path(panel_paths[L])
            if not path.exists():
                raise FileNotFoundError(path)
            src[L] = fitz.open(str(path))
        yield src
    finally:
        for doc in src.values():
            doc.close()

def _aspects(src: dict, letters: str) -> dict:
    return {L: src[L][0].rect.width / src[L][0].rect.height for L in letters}

def _letter_y(letter: str, rect: fitz.Rect, style: dict) -> float:
    overrides = style.get("letter_offsets_below_top", {})
    if letter in overrides:
        return rect.y0 + overrides[letter]
    fs = style["letter_fontsize"]
    return max(fs, rect.y0 - style["letter_offset_y"])

def _draw_letter(page: fitz.Page, letter: str, rect: fitz.Rect,
                 style: dict) -> None:
    page.insert_text(
        (rect.x0, _letter_y(letter, rect, style)),
        letter,
        fontsize=style["letter_fontsize"],
        fontname="hebo",
        color=style["letter_color"],
    )

def _render_and_save(page: fitz.Page, doc: fitz.Document, out_dir: Path,
                     name: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / f"{name}.pdf"
    doc.save(str(out_pdf), garbage=4, deflate=True)
    doc.close()

    check = fitz.open(str(out_pdf))
    pix = check[0].get_pixmap(dpi=300, alpha=False)
    out_png = out_dir / f"{name}.png"
    pix.save(str(out_png))
    n_images = len(check[0].get_images(full=True))
    pw, ph = check[0].rect.width, check[0].rect.height
    check.close()
    print(f"wrote {out_pdf}  page={pw:.1f} x {ph:.1f} pt  images={n_images}")
    return out_pdf, out_png

def _layout_figure_2(src: dict, style: dict) -> tuple[dict, float, float]:
    """Single column: A, B, C stacked at matched panel widths."""
    letters = style["letters"]
    A = _aspects(src, letters)
    margin_x, margin_y = style["margin"]
    _, gap_y = style["gap"]
    page_w = style["page_width"]
    usable_w = page_w - 2 * margin_x

    heights = {L: usable_w / A[L] for L in letters}
    body_h = sum(heights.values()) + gap_y * (len(letters) - 1)
    page_h = 2 * margin_y + body_h

    rects: dict = {}
    y = margin_y
    for L in letters:
        rects[L] = fitz.Rect(margin_x, y,
                             margin_x + usable_w, y + heights[L])
        y += heights[L] + gap_y
    return rects, page_w, page_h

def _layout_figure_3(src: dict, style: dict) -> tuple[dict, float, float]:
    """2 rows x 3 cols; row height solved so panels fill row width at native aspect."""
    A = _aspects(src, style["letters"])
    margin_x, margin_y = style["margin"]
    gap_x, gap_y = style["gap"]
    page_w = style["page_width"]
    usable_w = page_w - 2 * margin_x
    content_w = usable_w - 2 * gap_x

    row1_h = content_w / sum(A[L] for L in "ABC")
    row2_h = content_w / sum(A[L] for L in "DEF")
    page_h = 2 * margin_y + row1_h + gap_y + row2_h

    rects: dict = {}
    for row_letters, row_h, y0 in [
        ("ABC", row1_h, margin_y),
        ("DEF", row2_h, margin_y + row1_h + gap_y),
    ]:
        x = margin_x
        for L in row_letters:
            w = row_h * A[L]
            rects[L] = fitz.Rect(x, y0, x + w, y0 + row_h)
            x += w + gap_x
    return rects, page_w, page_h

def _layout_figure_4(src: dict, style: dict) -> tuple[dict, float, float]:
    """3 rows: A centred, [B/C stacked | D | E] middle, F | G bottom."""
    A = _aspects(src, style["letters"])
    margin_x, margin_y = style["margin"]
    gap_x, gap_y = style["gap"]
    bc_gap = style["bc_gap"]
    page_w = style["page_width"]
    usable_w = page_w - 2 * margin_x

    a_w = style["fixed_a_width"]
    a_h = a_w / A["A"]
    a_x = (page_w - a_w) / 2

    # Row height solved so column 1 (max(w_B, w_C)) + w_D + w_E + gaps
    # fills the usable width with B/C stacked.
    row2_w_avail = usable_w - 2 * gap_x
    a_bc = max(A["B"], A["C"])
    row2_h = (row2_w_avail + bc_gap * a_bc / 2) / (a_bc / 2 + A["D"] + A["E"])
    h_bc = (row2_h - bc_gap) / 2
    w_B = h_bc * A["B"]
    w_C = h_bc * A["C"]
    w_col1 = max(w_B, w_C)
    w_D = row2_h * A["D"]
    w_E = row2_h * A["E"]

    row2_y = margin_y + a_h + gap_y
    x_col2 = margin_x + w_col1 + gap_x
    x_col3 = x_col2 + w_D + gap_x

    row3_w_avail = usable_w - gap_x
    row3_h = row3_w_avail / (A["F"] + A["G"])
    w_F = row3_h * A["F"]
    w_G = row3_h * A["G"]
    row3_y = row2_y + row2_h + gap_y
    page_h = row3_y + row3_h + margin_y

    rects = {
        "A": fitz.Rect(a_x, margin_y, a_x + a_w, margin_y + a_h),
        "B": fitz.Rect(margin_x, row2_y, margin_x + w_B, row2_y + h_bc),
        "C": fitz.Rect(margin_x, row2_y + h_bc + bc_gap,
                       margin_x + w_C, row2_y + row2_h),
        "D": fitz.Rect(x_col2, row2_y, x_col2 + w_D, row2_y + row2_h),
        "E": fitz.Rect(x_col3, row2_y, x_col3 + w_E, row2_y + row2_h),
        "F": fitz.Rect(margin_x, row3_y, margin_x + w_F, row3_y + row3_h),
        "G": fitz.Rect(margin_x + w_F + gap_x, row3_y,
                       margin_x + w_F + gap_x + w_G, row3_y + row3_h),
    }
    return rects, page_w, page_h

def _layout_figure_S6(src: dict, style: dict) -> tuple[dict, float, float]:
    """2 rows: A centred, B | C below."""
    A = _aspects(src, style["letters"])
    margin_x, margin_y = style["margin"]
    gap_x, gap_y = style["gap"]
    page_w = style["page_width"]
    usable_w = page_w - 2 * margin_x

    a_w = style["fixed_a_width"]
    a_h = a_w / A["A"]
    a_x = (page_w - a_w) / 2

    row2_w_avail = usable_w - gap_x
    row2_h = row2_w_avail / (A["B"] + A["C"])
    w_B = row2_h * A["B"]
    w_C = row2_h * A["C"]

    row2_y = margin_y + a_h + gap_y
    page_h = row2_y + row2_h + margin_y

    rects = {
        "A": fitz.Rect(a_x, margin_y, a_x + a_w, margin_y + a_h),
        "B": fitz.Rect(margin_x, row2_y, margin_x + w_B, row2_y + row2_h),
        "C": fitz.Rect(margin_x + w_B + gap_x, row2_y,
                       margin_x + w_B + gap_x + w_C, row2_y + row2_h),
    }
    return rects, page_w, page_h

_LAYOUTS: dict[str, Callable] = {
    "figure_2":  _layout_figure_2,
    "figure_3":  _layout_figure_3,
    "figure_4":  _layout_figure_4,
    "figure_S6": _layout_figure_S6,
}

def compose(figure_name: str, panel_paths: dict,
            out_dir: Path) -> tuple[Path, Path]:
    """Stitch a named figure from panel PDFs into a vector composite."""
    if figure_name not in _LAYOUTS:
        raise KeyError(f"unknown figure {figure_name!r}; "
                       f"expected one of {sorted(_LAYOUTS)}")
    style = STYLES[figure_name]
    out_dir = Path(out_dir)
    with _open_sources(panel_paths, style["letters"]) as src:
        rects, page_w, page_h = _LAYOUTS[figure_name](src, style)
        doc = fitz.open()
        page = doc.new_page(width=page_w, height=page_h)
        for L in style["letters"]:
            page.show_pdf_page(rects[L], src[L], 0, keep_proportion=True)
            _draw_letter(page, L, rects[L], style)
        return _render_and_save(page, doc, out_dir, figure_name)

def compose_figure_2(panel_paths: dict, out_dir: Path) -> tuple[Path, Path]:
    return compose("figure_2", panel_paths, out_dir)

def compose_figure_3(panel_paths: dict, out_dir: Path) -> tuple[Path, Path]:
    return compose("figure_3", panel_paths, out_dir)

def compose_figure_4(panel_paths: dict, out_dir: Path) -> tuple[Path, Path]:
    return compose("figure_4", panel_paths, out_dir)

def compose_figure_S6(panel_paths: dict, out_dir: Path) -> tuple[Path, Path]:
    return compose("figure_S6", panel_paths, out_dir)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("figure", choices=sorted(_LAYOUTS))
    parser.add_argument("--out-dir", type=Path, required=True)
    for letter in "ABCDEFG":
        parser.add_argument(f"--panel-{letter.lower()}", type=Path,
                            default=None)
    args = parser.parse_args()

    style = STYLES[args.figure]
    panel_paths = {}
    for L in style["letters"]:
        p = getattr(args, f"panel_{L.lower()}")
        if p is None:
            parser.error(f"{args.figure} requires --panel-{L.lower()}")
        panel_paths[L] = p
    compose(args.figure, panel_paths, args.out_dir)

if __name__ == "__main__":
    main()
