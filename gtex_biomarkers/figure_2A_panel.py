"""Figure 2 Panel A — stacked-bar donor counts across tissue-pathology pairs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from matplotlib import rcParams

from gtex_biomarkers.config import Config
TABLE_CSV = Config.TABLES_DIR / "pair_class_balance.csv"
OUT_DIR = Config.FIGURES_DIR / "main_v2"

POS_COLOR = "#0072B2"
NEG_COLOR = "#E69F00"
BG_COLOR = "#FFFFFF"
BAND_COLOR = "#F0EFEB"
AXIS_COLOR = "#444444"
TICK_COLOR = "#333333"
LABEL_COLOR = "#333333"

TISSUE_ANCHOR_X = -0.28
TISSUE_ANCHOR_X_PER_COL = (-0.35, -0.50, -0.55)
TISSUE_LABEL_FONT_SIZE = 8
PATHOLOGY_LABEL_FONT_SIZE = 7
PATHOLOGY_LABEL_WEIGHT = "bold"
COUNT_LABEL_FONT_SIZE = 8
COUNT_LABEL_WEIGHT = "bold"
X_AXIS_LABEL_FONT_SIZE = 11
X_AXIS_LABEL_WEIGHT = "bold"
LEGEND_FONT_SIZE = 12

TISSUE_PATHOLOGY_GAP_PER_COL = (0.02, 0.00, 0.00)
BAR_HEIGHT_DATA = 0.50
BAND_EXTENT_FRAC = 0.42
Y_LINE_EXTENT_FRAC = 0.40
Y_LINE_COLOR = "#000000"
Y_LINE_WIDTH = 1.0

WSPACE_COL_1_2 = 0.60
WSPACE_COL_2_3 = 0.65

TISSUE_SPLIT = (slice(0, 4), slice(4, 10), slice(10, 20))

def _titlecase(s: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in s.replace("_", " ").split())

def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()
    d["tissue_pathology"] = d["tissue"] + " - " + d["pathology"]
    d["positive"] = d["n_pos"].astype(int)
    d["negative"] = d["n_neg"].astype(int)
    d["total"] = d["positive"] + d["negative"]
    d["path_lab"] = d["pathology"].map(_titlecase)

    tissue_maxtot = d.groupby("tissue")["total"].max().sort_values(ascending=False)
    tissue_order = tissue_maxtot.index.tolist()

    ordered = [d[d["tissue"] == t].sort_values("positive", ascending=False)
               for t in tissue_order]
    d_ord = pd.concat(ordered, ignore_index=True)
    return d_ord, tissue_order

def _xticks_step(gmax):
    return 200 if gmax > 500 else (100 if gmax > 250 else 50)

def _draw_column_content(ax, d_col, tissues_in_col, n_max_rows, show_xlabel,
                         tissue_anchor_x):
    """Draw bars, ticks, count text, and tissue strip labels for one column."""
    n_rows = len(d_col)
    d_col = d_col.reset_index(drop=True).copy()
    if n_rows > 1:
        y_positions = np.linspace(n_max_rows - 1, 0, n_rows)
        spacing = (n_max_rows - 1) / (n_rows - 1)
    else:
        y_positions = np.array([(n_max_rows - 1) / 2])
        spacing = 1.0
    d_col["y"] = y_positions
    band_extent = spacing * BAND_EXTENT_FRAC

    tissue_bounds = []
    for t in tissues_in_col:
        idx = d_col.index[d_col["tissue"] == t]
        if len(idx) > 0:
            tissue_bounds.append((t, idx.min(), idx.max()))

    gmax = d_col["total"].max()
    step = _xticks_step(gmax)
    xticks = np.arange(0, int(np.floor(gmax / step) * step) + step, step)
    x_left = -0.11 * gmax
    x_right = gmax * 1.09

    for xt in xticks:
        ax.axvline(xt, color="#E4E4E4", linewidth=0.35, zorder=1)

    for _, row in d_col.iterrows():
        ax.barh(row["y"], row["positive"], height=BAR_HEIGHT_DATA,
                color=POS_COLOR, edgecolor="none", zorder=2)
        ax.barh(row["y"], row["negative"], left=row["positive"],
                height=BAR_HEIGHT_DATA, color=NEG_COLOR, edgecolor="none",
                zorder=2)
        ax.text(-0.008 * gmax, row["y"], str(row["positive"]),
                ha="right", va="center",
                fontsize=COUNT_LABEL_FONT_SIZE, fontweight=COUNT_LABEL_WEIGHT,
                color=LABEL_COLOR, zorder=3)
        ax.text(row["total"] + 0.008 * gmax, row["y"],
                str(row["negative"]), ha="left", va="center",
                fontsize=COUNT_LABEL_FONT_SIZE, fontweight=COUNT_LABEL_WEIGHT,
                color=LABEL_COLOR, zorder=3)

    # Two vertical lines per tissue: (1) at x_left (panel edge), (2) at x=0 (bar start).
    line_extent = spacing * Y_LINE_EXTENT_FRAC
    for t, lo, hi in tissue_bounds:
        y_top_line = d_col.loc[lo, "y"] + line_extent
        y_bot_line = d_col.loc[hi, "y"] - line_extent
        ax.plot([x_left, x_left], [y_bot_line, y_top_line],
                color=Y_LINE_COLOR, linewidth=Y_LINE_WIDTH,
                solid_capstyle="butt", zorder=4, clip_on=False)
        ax.plot([0, 0], [y_bot_line, y_top_line],
                color=Y_LINE_COLOR, linewidth=Y_LINE_WIDTH,
                solid_capstyle="butt", zorder=4, clip_on=False)

    ax.set_yticks(d_col["y"].values)
    ax.set_yticklabels(d_col["path_lab"].values,
                       fontsize=PATHOLOGY_LABEL_FONT_SIZE,
                       fontweight=PATHOLOGY_LABEL_WEIGHT,
                       color=LABEL_COLOR)
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(-band_extent - 0.1, n_max_rows - 1 + band_extent + 0.1)
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", colors=TICK_COLOR, labelsize=8, length=3,
                   width=0.35)
    ax.tick_params(axis="y", colors="#666666", length=2, width=0.3)
    if show_xlabel:
        ax.set_xlabel("Number of donors",
                      fontsize=X_AXIS_LABEL_FONT_SIZE,
                      fontweight=X_AXIS_LABEL_WEIGHT,
                      color="#222222", labelpad=4)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_linewidth(0.5)

    tissue_label_texts = []
    for t, lo, hi in tissue_bounds:
        y_mid = (d_col.loc[lo, "y"] + d_col.loc[hi, "y"]) / 2
        label = t.replace(" - ", "\n")
        text = ax.text(tissue_anchor_x, y_mid, label,
                       ha="left", va="center", ma="left",
                       fontsize=TISSUE_LABEL_FONT_SIZE, color="#222222",
                       weight="bold", linespacing=1.15,
                       transform=ax.get_yaxis_transform(), clip_on=False)
        tissue_label_texts.append(text)

    ax.set_facecolor("none")
    return d_col, tissue_bounds, tissue_label_texts, band_extent

def _measure_leftmost_axes_x(ax, text_objs, renderer):
    """Return the leftmost x (axes fraction) across text objects, from rendered bboxes."""
    inv = ax.transAxes.inverted()
    leftmost = 0.0
    for txt in text_objs:
        bbox_disp = txt.get_window_extent(renderer=renderer)
        bbox_axes = bbox_disp.transformed(inv)
        if bbox_axes.x0 < leftmost:
            leftmost = bbox_axes.x0
    return leftmost

def _add_measured_bands(ax, d_col, tissue_bounds, band_x_start, band_extent):
    """Draw alternating grey bands starting from the measured leftmost x."""
    band_x_end = 1.0
    for band_i, (t, lo, hi) in enumerate(tissue_bounds):
        if band_i % 2 == 1:
            y_top = d_col.loc[lo, "y"] + band_extent
            y_bot = d_col.loc[hi, "y"] - band_extent
            rect = Rectangle(
                (band_x_start, y_bot),
                band_x_end - band_x_start,
                y_top - y_bot,
                facecolor=BAND_COLOR, edgecolor="none",
                transform=ax.get_yaxis_transform(),
                clip_on=False, zorder=0,
            )
            ax.add_patch(rect)

def build(df: pd.DataFrame,
          figsize: tuple[float, float] = (15.5, 7.2),
          band_x_pad: float = 0.02) -> plt.Figure:
    """Build the Fig 2A stacked-bar figure and return it."""
    d_ord, tissue_order = _prepare(df)

    tissue_groups = [tissue_order[s] for s in TISSUE_SPLIT]
    col_data = [d_ord[d_ord["tissue"].isin(g)].reset_index(drop=True)
                for g in tissue_groups]
    n_max_rows = max(len(c) for c in col_data)

    rcParams["font.family"] = "Arial"
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(BG_COLOR)
    gs = fig.add_gridspec(
        1, 5,
        width_ratios=[1.0, WSPACE_COL_1_2, 1.0, WSPACE_COL_2_3, 1.0],
        wspace=0.0, left=0.075, right=0.985, top=0.945, bottom=0.075,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 2, 4)]

    per_col_state = []
    for i, (ax, d_col, tissues) in enumerate(
        zip(axes, col_data, tissue_groups)
    ):
        state = _draw_column_content(
            ax, d_col, tissues, n_max_rows,
            show_xlabel=(i == 1),
            tissue_anchor_x=TISSUE_ANCHOR_X_PER_COL[i],
        )
        per_col_state.append((ax, state))

    # Band placement requires measured tissue-label bboxes → two-pass render.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for ax, (d_col, tissue_bounds, tissue_texts, band_extent) in per_col_state:
        leftmost = _measure_leftmost_axes_x(ax, tissue_texts, renderer)
        band_x_start = leftmost - band_x_pad
        _add_measured_bands(ax, d_col, tissue_bounds, band_x_start, band_extent)

    pos_patch = mpatches.Patch(facecolor=POS_COLOR,
                               label="Positive (has pathology)")
    neg_patch = mpatches.Patch(facecolor=NEG_COLOR, label="Negative")
    fig.legend(handles=[pos_patch, neg_patch],
               loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=2, frameon=False, fontsize=LEGEND_FONT_SIZE,
               handleheight=1.1, handlelength=1.5,
               columnspacing=2.4, borderaxespad=0)

    return fig

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TABLE_CSV)
    fig = build(df)
    out_pdf = OUT_DIR / "Figure_2A.pdf"
    out_png = OUT_DIR / "Figure_2A.png"
    fig.savefig(out_pdf, bbox_inches="tight", facecolor=BG_COLOR)
    fig.savefig(out_png, bbox_inches="tight", facecolor=BG_COLOR, dpi=500)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")

if __name__ == "__main__":
    main()
