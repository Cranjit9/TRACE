"""Shared style, palette, and helpers for the manuscript figures."""
from __future__ import annotations
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from trace_path.config import Config

STYLE = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         7,
    "axes.titlesize":    8,
    "axes.titleweight":  "bold",
    "axes.labelsize":    7,
    "axes.labelweight":  "normal",
    "axes.linewidth":    0.6,
    "axes.edgecolor":    "#333333",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.labelsize":   6.5,
    "ytick.labelsize":   6.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
    "legend.fontsize":   6.5,
    "legend.frameon":    False,
    "legend.handlelength": 1.2,
    "figure.dpi":        140,
    "savefig.dpi":       600,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
}

HOUSE_STYLE = {
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":           7.5,
    "axes.titlesize":      9,
    "axes.titleweight":    "bold",
    "axes.labelsize":      8,
    "axes.labelweight":    "normal",
    "axes.linewidth":      0.5,
    "axes.edgecolor":      "#333333",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.labelsize":     7,
    "ytick.labelsize":     7,
    "xtick.major.width":   0.5,
    "ytick.major.width":   0.5,
    "xtick.major.size":    2.5,
    "ytick.major.size":    2.5,
    "xtick.major.pad":     2.5,
    "ytick.major.pad":     2.5,
    "xtick.color":         "#333333",
    "ytick.color":         "#333333",
    "legend.fontsize":     7,
    "legend.frameon":      False,
    "legend.handlelength": 1.0,
    "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.3,
    "grid.color":          "#DDDDDD",
    "grid.linestyle":      "--",
    "grid.linewidth":      0.4,
    "grid.alpha":          0.6,
    "axes.grid":           False,
    "figure.dpi":          140,
    "savefig.dpi":         600,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.05,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
}

def apply_style(style=STYLE):
    mpl.rcParams.update(style)

def apply_house_style():
    mpl.rcParams.update(HOUSE_STYLE)

CORAL   = "#C44E52"
STEEL   = "#4878A8"
SAGE    = "#7B9E58"
GOLD    = "#D9B26F"
SLATE   = "#5C6B7A"
GRAY    = "#888888"
GRAY_LT = "#CCCCCC"

PRIMARY_BLUE   = "#5480AB"
DRIVER_YELLOW = "#D5A94A"
PROTECTIVE_PURPLE = "#7E6BA8"
HIGHLIGHT_RED    = "#B94A48"
NEUTRAL_GREY   = "#888888"
BAND_TAN   = "#F0EEE9"
BAND_STRIP = "#EAE5DE"
INK    = "#333333"
MUTED_GREY  = "#D9D9D9"

PALETTE = {
    "primary":     PRIMARY_BLUE,
    "driver":      DRIVER_YELLOW,
    "protective":  PROTECTIVE_PURPLE,
    "highlight":   HIGHLIGHT_RED,
    "grey":        NEUTRAL_GREY,
    "band":        BAND_TAN,
    "strip":       BAND_STRIP,
    "ink":         INK,
    "muted":       MUTED_GREY,
}

CMAP_AUC = LinearSegmentedColormap.from_list(
    "auc", ["#f7fbff", "#fee08b", "#fdae61", "#d73027", "#7f0000"], N=256)
CMAP_NES = LinearSegmentedColormap.from_list(
    "nes", ["#2166ac", "#d1e5f0", "#f7f7f7", "#fddbc7", "#b2182b"], N=256)
CMAP_RANK = LinearSegmentedColormap.from_list(
    "rank", ["#1a9850", "#a6d96a", "#fee08b", "#f46d43", "#a50026"], N=256)

CMAP_HOUSE_AUC = LinearSegmentedColormap.from_list(
    "house_auc",
    ["#3E6C9C", "#7FAAC7", "#C7DBEA",
     "#F5EFD8", "#F0D186", "#D48A4F",
     "#B34F35", "#8B2C1F"],
    N=256)

EN_DASH = "–"

def title_case_pathology(cat):
    """Title-case a pathology string, preserving known acronyms."""
    if not isinstance(cat, str):
        return cat
    ACRONYMS = {"NAFLD", "NASH", "DNA", "TF", "RNA", "GTEX", "MR"}
    parts = cat.replace("_", " ").split()
    out = []
    for p in parts:
        pu = p.upper()
        if pu in ACRONYMS:
            out.append(pu)
        else:
            out.append(p[:1].upper() + p[1:].lower())
    return " ".join(out)

def pretty_pair(tissue, category, wrap_small_intestine=False):
    """Format a tissue-pathology pair with en-dash separators."""
    t = str(tissue).strip().replace(" - ", f" {EN_DASH} ")
    c = title_case_pathology(str(category).strip())
    label = f"{t} {EN_DASH} {c}"
    if wrap_small_intestine and "Small Intestine" in label:
        label = label.replace(
            f"Small Intestine {EN_DASH} Terminal Ileum {EN_DASH} Nodularity",
            f"Small Intestine {EN_DASH}\nTerminal Ileum {EN_DASH}\nNodularity",
        )
    return label

TISSUE_DISPLAY = {
    "Adipose - Subcutaneous":        "Adipose\nSub",
    "Adipose - Visceral (Omentum)":  "Adipose\nVisc",
    "Artery - Aorta":                "Artery\nAorta",
    "Artery - Coronary":             "Artery\nCoronary",
    "Artery - Tibial":               "Artery\nTibial",
    "Breast - Mammary Tissue":       "Breast",
    "Esophagus - Mucosa":            "Esoph\nMuc",
    "Heart - Atrial Appendage":      "Heart\nAA",
    "Heart - Left Ventricle":        "Heart\nLV",
    "Kidney - Cortex":               "Kidney\nCtx",
    "Muscle - Skeletal":             "Muscle\nSk",
    "Small Intestine - Terminal Ileum": "Small Int.\nT. Ileum",
}

def tissue_display(tissue):
    return TISSUE_DISPLAY.get(tissue, tissue)

FINNGEN_LABEL = {
    "NAFLD":            "NAFLD",
    "NASH":             "NASH",
    "CIRRHOSIS_BROAD":  "CIRRHOSIS\n(BROAD)",
    "K11_FIBROCHIRLIV": "K11\nFIBROCHIRLIV",
    "CHIRHEP_NAS":      "CHRHEP\nNAS",
    "FIBROLIV":         "FIBROLIVER",
}

def panel_letter(ax, letter, dx=-0.13, dy=1.04, fontsize=10):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="bottom", ha="left")

def thin_spines(ax):
    for s in ax.spines.values():
        s.set_linewidth(0.6); s.set_color("#333333")
    return ax

def hs_axes(ax, spines=("left", "bottom"), gridx=False, gridy=False):
    """Normalise an axis to house style spines, ticks, and optional grid."""
    for name, s in ax.spines.items():
        if name in spines:
            s.set_visible(True)
            s.set_linewidth(0.5)
            s.set_color(INK)
        else:
            s.set_visible(False)
    ax.tick_params(axis="both", direction="out", length=2.5, width=0.5,
                   colors=INK, pad=2)
    if gridx:
        ax.grid(axis="x", color="#DDDDDD", linestyle="--",
                linewidth=0.4, alpha=0.7, zorder=0)
    if gridy:
        ax.grid(axis="y", color="#DDDDDD", linestyle="--",
                linewidth=0.4, alpha=0.7, zorder=0)
    return ax

def hs_panel_letter(fig, ax, letter, dx=-0.03, dy=0.01, fontsize=13):
    """Place a panel letter in figure-space above and left of the axis."""
    pos = ax.get_position()
    fig.text(pos.x0 + dx, pos.y1 + dy, letter,
             fontsize=fontsize, fontweight="bold", va="bottom", ha="left",
             color=INK)

def hs_errorbar(ax, x, y, xerr=None, yerr=None, color=None, **kw):
    """uncapped errorbar."""
    color = color if color is not None else PRIMARY_BLUE
    defaults = dict(fmt="o", markersize=6, markeredgecolor=INK,
                    markeredgewidth=0.4, ecolor=INK, elinewidth=0.6,
                    capsize=0, color=color)
    defaults.update(kw)
    return ax.errorbar(x, y, xerr=xerr, yerr=yerr, **defaults)

def hs_dumbbell(ax, y, x_left, x_right, *,
                  color_left=None, color_right=None,
                  bar_color=None, bar_lw=5, dot_size=(90, 110),
                  labels_delta=True, zorder_bar=1):
    """dumbbell row with tan connector and two colored dots."""
    color_left  = color_left  if color_left  is not None else DRIVER_YELLOW
    color_right = color_right if color_right is not None else PRIMARY_BLUE
    bar_color   = bar_color   if bar_color   is not None else "#E8CFA0"
    ax.plot([x_left, x_right], [y, y], color=bar_color, lw=bar_lw,
            solid_capstyle="round", zorder=zorder_bar)
    ax.scatter([x_left],  [y], s=dot_size[0], color=color_left,
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter([x_right], [y], s=dot_size[1], color=color_right,
               edgecolor="white", linewidth=0.6, zorder=3)
    if labels_delta:
        delta = x_right - x_left
        ax.annotate(f"+{delta:.2f}", xy=(x_right, y),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=HIGHLIGHT_RED, va="center")

def short_pair(t, c):
    """Compact tissue-pathology label."""
    t2 = (t.replace("Breast - Mammary Tissue", "Breast")
            .replace("Heart - Atrial Appendage", "Heart - AA")
            .replace("Heart - Left Ventricle",  "Heart - LV")
            .replace("Adipose - Subcutaneous",  "Adipose - Sub")
            .replace("Adipose - Visceral (Omentum)", "Adipose - Visc")
            .replace("Esophagus - Mucosa",      "Esoph - Muc")
            .replace("Muscle - Skeletal",       "Muscle - Sk")
            .replace("Kidney - Cortex",         "Kidney - Ctx"))
    return f"{t2} - {c}"

def save_fig(fig, name):
    """Save PDF + matching 600-dpi PNG into Config.FIGURES_DIR."""
    out = Config.FIGURES_DIR / name
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  saved {out.name}  ({out.stat().st_size/1024:.0f} KB)")

def load_data():
    """Load every CSV the figure cells reference, plus cohorts and PCA cum-var."""
    T = Config.TABLES_DIR
    data = {
        "lr":       pd.read_csv(T / "cv_results_all_tissue_pathology.csv"),
        "rf":       pd.read_csv(T / "cv_results_all_tissue_rf.csv"),
        "three":    pd.read_csv(T / "cv_three_way_comparison.csv"),
        "conf_imp": pd.read_csv(T / "confounder_feature_importances.csv"),
        "pc_auc":   pd.read_csv(T / "pc_auc_results.csv"),
        "pc_imp":   pd.read_csv(T / "pc_gene_importance_full.csv"),
        "pc_vs_rf": pd.read_csv(T / "validation_pc_vs_rf.csv"),
        "gsea":     pd.read_csv(T / "gsea_pathway_enrichment.csv"),
    }
    data["pc_auc_top"] = (
        data["pc_auc"][(data["pc_auc"]["auc_pc_conf"] >= Config.AUC_THRESH) &
                       (data["pc_auc"]["delta_pc"]    >= Config.DELTA_THRESH)]
            .sort_values("delta_pc", ascending=False).reset_index(drop=True))
    data["three_top"] = (
        data["three"][(data["three"]["auc_comb"] >= Config.AUC_THRESH) &
                      (data["three"]["delta"]    >= Config.DELTA_THRESH)]
            .sort_values("delta", ascending=False).reset_index(drop=True))

    pca_csv = T / "pca_cumulative_variance_wb20k.csv"
    if not pca_csv.exists():
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from trace_path.data import variance_filter
        with open(Config.CACHE_DIR / "processed_data.pkl", "rb") as f:
            proc = pickle.load(f)
        X_var, _ = variance_filter(proc["X_wb"])
        X_scaled = StandardScaler().fit_transform(X_var)
        n_comp = min(800, X_scaled.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=Config.SEED).fit(X_scaled)
        pd.DataFrame({"pc": np.arange(1, n_comp + 1),
                      "cum_var": np.cumsum(pca.explained_variance_ratio_)}
                     ).to_csv(pca_csv, index=False)
    data["pca_cv"] = pd.read_csv(pca_csv)
    return data
