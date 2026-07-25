"""Builders for supplementary figures S1-S6."""
from __future__ import annotations
import ast
import io
from itertools import combinations
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import (
    FancyArrowPatch, FancyBboxPatch, Rectangle, Patch,
)
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

from trace_path.figure_builder import (
    apply_style, panel_letter, thin_spines, short_pair,
    CORAL, STEEL, SAGE, GOLD, SLATE, GRAY, GRAY_LT,
    CMAP_AUC, CMAP_NES, CMAP_RANK,
)
from trace_path.config import Config
from trace_path.data import (
    load_cache, build_confounder_matrix, variance_filter,
)
from trace_path.labels import assign_donor_labels
from trace_path.models import (
    make_rf_model, _auc_feature_selection,
    _compute_pc_covariate_aucs, _drop_covariate_aligned_pcs,
)

TABLES = Config.TABLES_DIR

def _fix_pair(label):
    """Convert 'Liver | cirrhosis' -> 'Liver - cirrhosis'."""
    if not isinstance(label, str):
        return label
    return label.replace(" | ", " - ").replace("|", " - ")

def _wrap_si(lbl):
    """Break Small Intestine label onto 3 lines (Nature-minimalist convention)."""
    return lbl.replace(
        "Small Intestine - Terminal Ileum - nodularity",
        "Small Intestine -\nTerminal Ileum -\nnodularity",
    )

def _qual_pairs():
    """Recompute qualifying-pair list from live pc_auc + Config thresholds."""
    pc_auc = pd.read_csv(TABLES / "pc_auc_results.csv")
    mask = (pc_auc["auc_pc_conf"] >= Config.AUC_THRESH) & (pc_auc["delta_pc"] >= Config.DELTA_THRESH)
    return pc_auc.loc[mask, ["tissue", "category"]].values.tolist()

def _display_pair(t, c):
    return f"{t} - {c}"

from pathlib import Path

import matplotlib.pyplot as plt

pca_cv = pd.read_csv(TABLES / "pca_cumulative_variance_wb20k.csv")
donor  = pd.read_csv(TABLES / "pca_donor_scores_wb20k.csv")
pc_auc = pd.read_csv(TABLES / "pc_auc_results.csv")
pair_bal = pd.read_csv(TABLES / "pair_class_balance.csv")
dropped = pd.read_csv(TABLES / "covariate_dropped_pcs.csv")
sweep = {t: pd.read_csv(TABLES / f"pc_auc_results_thresh_{t:.2f}.csv")
         for t in (0.65, 0.70, 0.75)}

qual_mask = (pc_auc["auc_pc_conf"] >= Config.AUC_THRESH) & (pc_auc["delta_pc"] >= Config.DELTA_THRESH)
qual_pairs = pc_auc.loc[qual_mask, ["tissue", "category"]].values.tolist()
qual_set = {(t, c) for t, c in qual_pairs}
print(f"Qualifying pairs: {len(qual_pairs)}")
for t, c in qual_pairs:
    print(f"  - {t} - {c}")

def display_pair(tissue: str, cat: str) -> str:
    """Return 'Tissue - pathology' (spaced hyphen), Nature-style."""
    return f"{tissue} - {cat}"

def build_S1():
    """Build Fig S1 — PCA variance decomposition + donor covariate distributions."""
    apply_style()
    fig = plt.figure(figsize=(9.0, 5.4))
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.55, wspace=1.3)
    axA = fig.add_subplot(gs[0, 0:3])
    axB = fig.add_subplot(gs[0, 3:6])
    axC = fig.add_subplot(gs[1, 0:2])
    axD = fig.add_subplot(gs[1, 2:4])
    axE = fig.add_subplot(gs[1, 4:6])

    cum_pct = pca_cv["cum_var"].values * 100
    pcs = pca_cv["pc"].values
    axA.plot(pcs, cum_pct, color=STEEL, lw=1.2)
    axA.axhline(80, color=GRAY, ls=":", lw=0.6)
    axA.axhline(90, color=GRAY, ls=":", lw=0.6)
    pc100_val = cum_pct[99] if len(cum_pct) >= 100 else cum_pct[-1]
    axA.axvline(100, color=CORAL, ls="--", lw=0.8)
    axA.text(120, pc100_val - 8, f"Top 100 selected,\n{pc100_val:.1f}% variance",
             color=CORAL, fontsize=6.5, ha="left", va="top")
    axA.text(pcs.max(), 80.5, "80%", color=GRAY, fontsize=6, ha="right", va="bottom")
    axA.text(pcs.max(), 90.5, "90%", color=GRAY, fontsize=6, ha="right", va="bottom")
    axA.set_xlabel("Principal component")
    axA.set_ylabel("Cumulative variance (%)")
    axA.set_xlim(0, pcs.max())
    axA.set_ylim(0, 100)
    thin_spines(axA)
    panel_letter(axA, "A")

    var_pct = pca_cv["cum_var"].diff().fillna(pca_cv["cum_var"].iloc[0]) * 100
    top50 = min(50, len(var_pct))
    axB.bar(range(1, top50 + 1), var_pct.iloc[:top50], color=STEEL, width=0.85,
            edgecolor="none")
    axB.set_xlabel("Principal component")
    axB.set_ylabel("Variance explained (%)")
    axB.set_xlim(0.2, top50 + 0.8)
    thin_spines(axB)
    panel_letter(axB, "B")

    age = donor["AGE"].dropna()
    axC.hist(age, bins=np.arange(20, 75, 5), color=SAGE, edgecolor="white", linewidth=0.4)
    axC.set_xlabel("Age (years)")
    axC.set_ylabel("Donors")
    axC.text(0.02, 0.95, f"n = {len(age)}", transform=axC.transAxes,
             ha="left", va="top", fontsize=6.5, color=SLATE)
    axC.set_xlim(18, 72)
    thin_spines(axC)
    panel_letter(axC, "C")

    hardy = donor["DTHHRDY"].dropna().astype(int)
    counts = hardy.value_counts().sort_index()
    # Ensure 0..4 present
    for lvl in range(5):
        if lvl not in counts.index:
            counts.loc[lvl] = 0
    counts = counts.sort_index()
    axD.bar(counts.index.astype(str), counts.values, color=GOLD, width=0.8,
            edgecolor="none")
    axD.set_xlabel("Hardy scale")
    axD.set_ylabel("Donors")
    axD.text(0.98, 0.95, f"n = {int(counts.sum())}", transform=axD.transAxes,
             ha="right", va="top", fontsize=6.5, color=SLATE)
    thin_spines(axD)
    panel_letter(axD, "D")

    isch = donor["TRISCHD"].dropna()
    axE.hist(isch, bins=np.arange(0, isch.max() + 100, 100), color=CORAL,
             edgecolor="white", linewidth=0.4)
    axE.set_xlabel("Total ischemic time (min)")
    axE.set_ylabel("Donors")
    axE.text(0.98, 0.95, f"n = {len(isch)}", transform=axE.transAxes,
             ha="right", va="top", fontsize=6.5, color=SLATE)
    thin_spines(axE)
    panel_letter(axE, "E")
    return fig

def compute_S2A_matrix():
    """Regenerate top-50 PC x 5-covariate AUC matrix for Liver-cirrhosis fold 1."""
    import pickle
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedGroupKFold

    from trace_path.data import variance_filter, build_confounder_matrix
    from trace_path.labels import assign_donor_labels
    from trace_path.models import _compute_pc_covariate_aucs

    with open(Config.CACHE_DIR / "processed_data.pkl", "rb") as f:
        proc = pickle.load(f)
    X_wb = proc["X_wb"]
    blood_subjid = proc["blood_subjid"]
    df_meta_url = proc["df_meta_url"]
    df_age = proc["df_age"]

    X_var, _ = variance_filter(X_wb)
    X_conf_all = build_confounder_matrix(df_age, blood_subjid)

    y, _, _, _ = assign_donor_labels(df_meta_url, "Liver", "cirrhosis", blood_subjid)
    keep = y.notna()
    y_clean = y[keep].astype(int)
    g = blood_subjid[keep].astype(str)
    X_sub = X_var.loc[keep]
    X_conf_sub = X_conf_all.loc[keep]

    cv = StratifiedGroupKFold(
        n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED
    )
    splits = list(cv.split(X_sub.values, y_clean.values, groups=g.values))
    tr_idx, te_idx = splits[0]
    Xtr = X_sub.iloc[tr_idx]

    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr)
    n_comp = min(800, Xtr_scaled.shape[0] - 1, Xtr_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=Config.SEED).fit(Xtr_scaled)
    scores_tr = pca.transform(Xtr_scaled)
    pc_cols = [f"PC{i+1}" for i in range(n_comp)]
    pcs_df = pd.DataFrame(scores_tr, columns=pc_cols, index=Xtr.index)
    X_conf_tr = X_conf_sub.iloc[tr_idx]

    aucs = _compute_pc_covariate_aucs(pcs_df, X_conf_tr)
    return aucs.head(50)

def build_S2():
    """Build Fig S2 — covariate-orthogonal PC screening pipeline."""
    apply_style()

    fig = plt.figure(figsize=(11.5, 13.5))
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[0.75, 1.35, 0.85],
        hspace=0.55, wspace=0.35,
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axHM = fig.add_subplot(gs[1, :])
    axD = fig.add_subplot(gs[2, 0])
    axE = fig.add_subplot(gs[2, 1])

    thresholds = [0.65, 0.70, 0.75]
    pair_colors = [CORAL, STEEL, SAGE, GOLD]
    pair_labels_short = []
    for (t_, c_), color in zip(qual_pairs, pair_colors):
        pair_labels_short.append((t_, c_, color))

    xs = np.arange(len(thresholds))
    for (t_, c_, color) in pair_labels_short:
        ys = []
        for thr in thresholds:
            df = sweep[thr]
            row = df[(df["tissue"] == t_) & (df["category"] == c_)]
            if len(row):
                ys.append(row["auc_pc_conf"].iloc[0])
            else:
                ys.append(np.nan)
        axA.plot(xs, ys, marker="o", color=color, lw=1.1, ms=4,
                 label=display_pair(t_, c_))
    axA.axhline(Config.AUC_THRESH, color=GRAY, ls=":", lw=0.6)
    axA.text(len(thresholds) - 0.9, Config.AUC_THRESH + 0.005,
             "AUC gate",
             fontsize=6, color=GRAY, ha="right", va="bottom")
    # Line-end labels (colour-matched to pair) act as inline legend; suppress the boxed legend
    def _wrap_small_intestine(lbl: str) -> str:
        return lbl.replace(
            "Small Intestine - Terminal Ileum - nodularity",
            "Small Intestine -\nTerminal Ileum -\nnodularity",
        )
    for (t_, c_, color) in pair_labels_short:
        df = sweep[thresholds[-1]]
        row = df[(df["tissue"] == t_) & (df["category"] == c_)]
        if len(row):
            y_end = row["auc_pc_conf"].iloc[0]
            axA.text(len(thresholds) - 1 + 0.06, y_end,
                     _wrap_small_intestine(display_pair(t_, c_)),
                     fontsize=5.8, color=color,
                     ha="left", va="center")
    axA.set_xticks(xs)
    axA.set_xticklabels([f"{t:.2f}" for t in thresholds])
    axA.set_xlabel("Orthogonality threshold")
    axA.set_ylabel("PC + covariates AUC")
    axA.set_xlim(-0.15, len(thresholds) - 1 + 1.5)  # room for line-end labels
    thin_spines(axA)
    panel_letter(axA, "A")

    drop70 = dropped[dropped["threshold"] == 0.70]

    all_vals = drop70["n_dropped"].values
    axB.boxplot(
        [all_vals], vert=True, widths=0.55, positions=[0],
        patch_artist=True,
        boxprops=dict(facecolor=GRAY_LT, edgecolor=SLATE, lw=0.6),
        medianprops=dict(color=SLATE, lw=1.0),
        whiskerprops=dict(color=SLATE, lw=0.6),
        capprops=dict(color=SLATE, lw=0.6),
        flierprops=dict(marker="o", markersize=1.8, markerfacecolor=GRAY,
                        markeredgecolor="none", alpha=0.5),
    )
    rng = np.random.default_rng(0)
    # Widened spacing so 3-line Small Intestine label does not overlap Lung - congestion
    x_positions = [1.5, 3.1, 4.9, 6.7]
    for (t_, c_), color, xp in zip(qual_pairs, pair_colors, x_positions):
        vals = drop70[(drop70["tissue"] == t_) & (drop70["category"] == c_)]["n_dropped"].values
        jitter = rng.normal(0, 0.08, size=len(vals))
        axB.scatter(xp + jitter, vals, s=22, color=color,
                    edgecolor="white", lw=0.4, zorder=5)
        axB.plot([xp - 0.22, xp + 0.22], [vals.mean(), vals.mean()],
                 color=color, lw=1.0, solid_capstyle="round")

    axB.set_xticks([0] + x_positions)
    xtick_lbls = ["All 59 pairs"] + [
        _wrap_small_intestine(display_pair(t_, c_)) for t_, c_ in qual_pairs
    ]
    axB.set_xticklabels(xtick_lbls, rotation=25, ha="right", fontsize=5.2)
    # Colour qualifying-pair x-tick labels to match Panel A lines (shared legend)
    for tick, (_, _), color in zip(axB.get_xticklabels()[1:], qual_pairs, pair_colors):
        tick.set_color(color)
    axB.set_ylabel("PCs dropped (out of 800)")
    axB.set_xlim(-0.6, 7.3)
    thin_spines(axB)
    panel_letter(axB, "B")

    piv = pc_auc.pivot(index="tissue", columns="category", values="auc_pc_conf")
    tissue_order = piv.max(axis=1).sort_values(ascending=False).index.tolist()
    piv = piv.loc[tissue_order]
    path_order = sorted(piv.columns)
    piv = piv[path_order]

    im = axHM.imshow(piv.values, aspect="auto", cmap=CMAP_AUC,
                     vmin=0.45, vmax=0.85, interpolation="nearest")
    axHM.set_xticks(range(len(piv.columns)))
    axHM.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=5.5)
    axHM.set_yticks(range(len(piv.index)))
    axHM.set_yticklabels(piv.index, fontsize=5.5)
    for (t_, c_) in qual_pairs:
        if t_ in piv.index and c_ in piv.columns:
            i = piv.index.get_loc(t_)
            j = piv.columns.get_loc(c_)
            axHM.add_patch(Rectangle(
                (j - 0.5, i - 0.5), 1, 1, fill=False,
                edgecolor=CORAL, lw=1.4,
            ))
    cbar = fig.colorbar(im, ax=axHM, fraction=0.02, pad=0.02)
    cbar.set_label("PC + covariates AUC", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)
    thin_spines(axHM)
    panel_letter(axHM, "C", dx=-0.06, dy=1.01)

    ranked = pc_auc.copy()
    ranked["display"] = ranked.apply(lambda r: display_pair(r["tissue"], r["category"]), axis=1)
    ranked["is_qual"] = ranked.apply(
        lambda r: (r["tissue"], r["category"]) in qual_set, axis=1
    )
    ranked = ranked.sort_values("auc_pc_conf", ascending=False).reset_index(drop=True)
    x = np.arange(len(ranked))
    colors_c = [CORAL if q else GRAY_LT for q in ranked["is_qual"]]
    axD.bar(x, ranked["auc_pc_conf"], color=colors_c, edgecolor="none", width=0.8)
    axD.axhline(0.60, color=GRAY, ls=":", lw=0.6)
    axD.axhline(0.70, color=GRAY, ls=":", lw=0.6)
    axD.text(len(ranked) - 1, 0.605, "AUC gate",
             color=GRAY, fontsize=6, ha="right", va="bottom")
    axD.text(len(ranked) - 1, 0.705, "Strong discrimination",
             color=GRAY, fontsize=6, ha="right", va="bottom")
    axD.set_xticks([])
    axD.set_xlabel(f"Tissue - pathology pairs ranked (n = {len(ranked)})")
    axD.set_ylabel("PC + covariates AUC")
    axD.set_ylim(0.45, max(0.85, ranked["auc_pc_conf"].max() + 0.03))
    for i, row in ranked.iterrows():
        if row["is_qual"]:
            axD.text(i, row["auc_pc_conf"] + 0.005,
                     display_pair(row["tissue"], row["category"]),
                     rotation=90, va="bottom", ha="center",
                     fontsize=5.5, color=CORAL)
    thin_spines(axD)
    panel_letter(axD, "D")

    stab = pc_auc.copy()
    stab["is_qual"] = stab.apply(
        lambda r: (r["tissue"], r["category"]) in qual_set, axis=1
    )
    nonq = stab[~stab["is_qual"]]
    q = stab[stab["is_qual"]]
    axE.scatter(nonq["auc_pc_conf"], nonq["std_pc_conf"],
                color=GRAY_LT, edgecolor=GRAY, lw=0.4, s=22, label="Non-qualifying")
    axE.scatter(q["auc_pc_conf"], q["std_pc_conf"],
                color=CORAL, edgecolor="white", lw=0.5, s=42, label="Qualifying (n = 4)",
                zorder=5)
    for _, r in q.iterrows():
        axE.annotate(
            display_pair(r["tissue"], r["category"]),
            xy=(r["auc_pc_conf"], r["std_pc_conf"]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=6, color=CORAL,
        )
    med_std = stab["std_pc_conf"].median()
    axE.axhline(med_std, color=GRAY, ls=":", lw=0.6)
    axE.axvline(0.60, color=GRAY, ls=":", lw=0.6)
    axE.set_xlabel("Mean PC + covariates AUC")
    axE.set_ylabel("Fold AUC standard deviation")
    axE.legend(fontsize=6, loc="upper left", handlelength=1.2, borderaxespad=0.3)
    thin_spines(axE)
    panel_letter(axE, "E")

    return fig

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def fix_pair(label: str) -> str:
    """Convert 'Liver | cirrhosis' -> 'Liver - cirrhosis'; also normalize."""
    if not isinstance(label, str):
        return label
    return label.replace(" | ", " - ").replace("|", " - ")

def build_S3():
    """Build Fig S3 — model-comparison panels for the 59 pairs and the 4 qualifying pairs."""
    apply_style()
    fig = plt.figure(figsize=(7.4, 8.2))
    # Nested GridSpec so A stays generous (needs room for A's 59 vertical x-labels) while B-C is tighter.
    outer = GridSpec(2, 1, figure=fig, hspace=0.9,
                     height_ratios=[1.25, 1.85])
    axA_gs = outer[0, 0]
    inner = outer[1, 0].subgridspec(2, 2, hspace=0.55, wspace=0.40,
                                     height_ratios=[1.0, 1.0])

    ax = fig.add_subplot(axA_gs)
    three = pd.read_csv(TABLES / "cv_three_way_comparison.csv")
    three["label"] = [short_pair(t, c) for t, c in
                       zip(three["tissue"], three["category"])]
    # Keep Small Intestine single-line in Panel A only: at 90 rotation with 59 densely packed labels, multi-line makes the label 3x wider and collides with neighbours.
    three["label"] = three["label"].map(fix_pair)
    three = three.sort_values("auc_comb", ascending=False).reset_index(drop=True)
    n = len(three)
    x = np.arange(n)
    w = 0.28
    ax.bar(x - w, three["auc_conf"], w, color=GRAY_LT,
           edgecolor="#555", linewidth=0.3, label="Covariates only")
    ax.bar(x,     three["auc_expr"], w, color=STEEL,
           edgecolor="none", label="Expression only")
    ax.bar(x + w, three["auc_comb"], w, color=CORAL,
           edgecolor="none", label="Expression + covariates")
    ax.axhline(0.60, color="#333", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(three["label"], rotation=90, fontsize=4.8,
                       ha="right", va="top", rotation_mode="anchor")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.7, n - 0.3)
    # Legend above the panel, clear of the bars
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, fontsize=6, handlelength=0.9,
              columnspacing=0.8, frameon=False)
    thin_spines(ax)
    panel_letter(ax, "A", dx=-0.06, dy=1.03)

    ax = fig.add_subplot(inner[0, :])
    m = pd.read_csv(TABLES / "six_model_auc_matrix_mean.csv")
    s = pd.read_csv(TABLES / "six_model_auc_matrix_std.csv")
    m["pathology"] = m["pathology"].map(fix_pair)
    s["pathology"] = s["pathology"].map(fix_pair)
    models = ["LR raw", "RF raw", "PC-LR (expr)", "PC-RF (expr)",
              "PC-LR (expr+covars)", "PC-RF (expr+covars)"]
    pair_colors = {
        "Liver - cirrhosis": CORAL,
        "Liver - steatosis": STEEL,
        "Small Intestine - Terminal Ileum - nodularity": SAGE,
        "Lung - congestion": GOLD,
    }
    n_pairs = len(m)
    bw = 0.18
    x = np.arange(len(models))
    for i, row in m.iterrows():
        pair = row["pathology"]
        vals = row[models].values.astype(float)
        errs = s[s["pathology"] == pair][models].values.astype(float).ravel()
        color = pair_colors.get(pair, GRAY)
        offset = (i - (n_pairs - 1) / 2) * bw
        display_pair_label = pair.replace(
            "Small Intestine - Terminal Ileum - nodularity",
            "Small Intestine -\nTerminal Ileum -\nnodularity",
        )
        ax.bar(x + offset, vals, bw, yerr=errs,
               color=color, edgecolor="none",
               error_kw={"elinewidth": 0.5, "capsize": 1.5, "ecolor": "#333"},
               label=display_pair_label)
    liver_row = m[m["pathology"] == "Liver - cirrhosis"].iloc[0]
    i_liver = m.index[m["pathology"] == "Liver - cirrhosis"][0]
    offset_liver = (i_liver - (n_pairs - 1) / 2) * bw
    ax.plot(x + offset_liver, liver_row[models].values.astype(float),
            color="#7a1a24", linewidth=0.7, marker="o", markersize=2.5,
            markerfacecolor="#7a1a24", markeredgecolor="white",
            markeredgewidth=0.3, zorder=5)
    ax.axhline(0.60, color="#333", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=6)
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.0)
    # Legend above the panel, clear of the bars
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              fontsize=5.8, ncol=4, handlelength=0.9,
              columnspacing=0.8, frameon=False)
    thin_spines(ax)
    panel_letter(ax, "B", dx=-0.06, dy=1.03)

    ax = fig.add_subplot(inner[1, 0])
    lrrf = pd.read_csv(TABLES / "comparison_lr_vs_rf.csv")
    lrrf["label"] = (lrrf["tissue"] + " - " + lrrf["category"]).map(fix_pair)
    qualifying = {
        "Liver - cirrhosis", "Liver - steatosis",
        "Small Intestine - Terminal Ileum - nodularity", "Lung - congestion",
    }
    is_qual = lrrf["label"].isin(qualifying)
    ax.scatter(lrrf.loc[~is_qual, "mean_auc_lr"],
               lrrf.loc[~is_qual, "mean_auc_rf"],
               s=14, color=GRAY, alpha=0.6, edgecolor="none")
    ax.scatter(lrrf.loc[is_qual, "mean_auc_lr"],
               lrrf.loc[is_qual, "mean_auc_rf"],
               s=30, color=CORAL, edgecolor="#333", linewidth=0.4, zorder=4)
    # Modest offsets, all in positive dy (labels above points) so pointer
    # arrows travel upward — visually consistent.
    label_offsets = {
        "Liver - cirrhosis":                                  ( 10, 12),
        "Liver - steatosis":                                  (-14, 12),
        "Small Intestine - Terminal Ileum - nodularity":      ( -8, 18),
        "Lung - congestion":                                  ( 14, 10),
    }
    for _, r in lrrf[is_qual].iterrows():
        short = r["label"].replace("Small Intestine - Terminal Ileum - nodularity",
                                    "Small Intestine -\nTerminal Ileum -\nnodularity")
        dx_off, dy_off = label_offsets.get(r["label"], (4, 4))
        ha = "left" if dx_off >= 0 else "right"
        ax.annotate(short, (r["mean_auc_lr"], r["mean_auc_rf"]),
                    xytext=(dx_off, dy_off), textcoords="offset points",
                    fontsize=5.5, color="#333", ha=ha,
                    arrowprops=dict(arrowstyle="-", color=GRAY,
                                    linewidth=0.3, shrinkA=0, shrinkB=2))
    lim = [0.3, 1.0]
    ax.plot(lim, lim, color="#333", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("LR AUC")
    ax.set_ylabel("RF AUC")
    thin_spines(ax)
    panel_letter(ax, "C", dx=-0.20, dy=1.06)

    axD = fig.add_subplot(inner[1, 1])
    six = pd.read_csv(TABLES / "liver_cirrhosis_six_model_auc.csv")
    y = np.arange(len(six))
    colours = [SLATE, SLATE, STEEL, STEEL, CORAL, CORAL]
    axD.barh(y, six["mean_auc"], xerr=six["std_auc"],
             color=colours, edgecolor="#333", linewidth=0.4,
             error_kw={"elinewidth": 0.5, "capsize": 1.5, "ecolor": "#333"})
    axD.axvline(0.60, color="#333", linestyle="--", linewidth=0.5, alpha=0.7)
    axD.set_yticks(y)
    axD.set_yticklabels(six["model"], fontsize=6)
    axD.invert_yaxis()
    axD.set_xlim(0, 1.0)
    axD.set_xlabel("AUC (Liver - cirrhosis)")
    for i, (m_, s_) in enumerate(zip(six["mean_auc"], six["std_auc"])):
        axD.text(m_ + s_ + 0.015, i, f"{m_:.2f}", va="center",
                 ha="left", fontsize=5.5)
    thin_spines(axD)
    panel_letter(axD, "D", dx=-0.35, dy=1.06)

    return fig

def build_S5():
    """Build Fig S5 — biological interpretation heatmaps for the 4 qualifying pairs."""
    apply_style()
    fig = plt.figure(figsize=(9.5, 10.5))
    gs = GridSpec(
        2, 2, figure=fig,
        height_ratios=[0.55, 1.20],
        width_ratios=[1.0, 1.15],
        hspace=0.55, wspace=0.60,
    )

    axA = fig.add_subplot(gs[0, 0])
    ss = pd.read_csv(TABLES / "signature_specificity_matrix.csv", index_col=0)
    ss.index = [fix_pair(p) for p in ss.index]
    ss.columns = [fix_pair(p) for p in ss.columns]
    order = ["Liver - cirrhosis", "Liver - steatosis",
             "Small Intestine - Terminal Ileum - nodularity",
             "Lung - congestion"]
    ss = ss.loc[order, order]
    im = axA.imshow(ss.values, cmap=CMAP_AUC, aspect="auto",
                    vmin=0.4, vmax=0.85)
    axA.set_xticks(range(len(order)))
    axA.set_yticks(range(len(order)))
    labels_short = [p.replace("Small Intestine - Terminal Ileum - nodularity",
                              "Small Intestine -\nTerminal Ileum -\nnodularity") for p in order]
    axA.set_xticklabels(labels_short, rotation=35, ha="right", fontsize=6)
    axA.set_yticklabels(labels_short, fontsize=6)
    for i in range(len(order)):
        for j in range(len(order)):
            val = ss.values[i, j]
            weight = "bold" if i == j else "normal"  # bold diagonals only
            axA.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=6, color="black", fontweight=weight)
        axA.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1,
                                fill=False, edgecolor="black",
                                linewidth=1.2, zorder=5))
    cbar = fig.colorbar(im, ax=axA, fraction=0.045, pad=0.04)
    cbar.set_label("Cross-pair AUC", fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    thin_spines(axA)
    panel_letter(axA, "A", dx=-0.30, dy=1.03)

    axB = fig.add_subplot(gs[0, 1])
    g = pd.read_csv(TABLES / "cross_organ_plasma_gsea_full.csv")
    g["pair"] = g["pair"].map(fix_pair)
    pair_order = order
    organ_order = sorted(g["organ_set"].unique())
    nes_mat = (g.pivot(index="pair", columns="organ_set", values="NES")
                .reindex(index=pair_order, columns=organ_order))
    vmax = float(np.nanmax(np.abs(nes_mat.values)))
    im = axB.imshow(nes_mat.values, cmap=CMAP_NES, aspect="auto",
                    vmin=-vmax, vmax=vmax)
    axB.set_xticks(range(len(organ_order)))
    axB.set_xticklabels(organ_order, rotation=35, ha="right", fontsize=6)
    axB.set_yticks(range(len(pair_order)))
    axB.set_yticklabels(
        [p.replace("Small Intestine - Terminal Ileum - nodularity",
                   "Small Intestine -\nTerminal Ileum -\nnodularity")
         for p in pair_order],
        fontsize=6)
    axB.set_xlabel("Plasma organ", fontsize=6.5)
    # Own-organ mapping (bold weight applies to own-organ cells)
    own_map = {
        "Liver - cirrhosis": "Liver",
        "Liver - steatosis": "Liver",
        "Small Intestine - Terminal Ileum - nodularity": "Intestine",
        "Lung - congestion": "Lung",
    }
    own_cell_set = {(i, organ_order.index(own_map[pair]))
                    for i, pair in enumerate(pair_order)
                    if own_map[pair] in organ_order}
    for i in range(nes_mat.shape[0]):
        for j in range(nes_mat.shape[1]):
            val = nes_mat.values[i, j]
            if np.isnan(val):
                continue
            weight = "bold" if (i, j) in own_cell_set else "normal"
            axB.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=5.5, color="black", fontweight=weight)
    for (i, j) in own_cell_set:
        axB.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                fill=False, edgecolor="black",
                                linewidth=1.2, zorder=5))
    cbar = fig.colorbar(im, ax=axB, fraction=0.045, pad=0.04)
    cbar.set_label("NES", fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    thin_spines(axB)
    panel_letter(axB, "B", dx=-0.22, dy=1.03)

    axC = fig.add_subplot(gs[1, :])

    _gsea = pd.read_csv(TABLES / "gsea_pathway_enrichment.csv")
    _LABEL_MAP = {
        "Liver | cirrhosis":                              "Liver - cirrhosis",
        "Liver | steatosis":                              "Liver - steatosis",
        "Lung | congestion":                              "Lung - congestion",
        "Small Intestine - Terminal Ileum | nodularity":  "Small Intestine - Terminal Ileum - nodularity",
    }
    _gsea["display"] = _gsea["label"].map(_LABEL_MAP)
    _SHORT_LABEL = {
        "Liver - cirrhosis": "Liver -\ncirrhosis",
        "Liver - steatosis": "Liver -\nsteatosis",
        "Lung - congestion": "Lung -\ncongestion",
        "Small Intestine - Terminal Ileum - nodularity":
            "Small Intestine -\nTerminal Ileum -\nnodularity",
    }
    _sub = _gsea[(_gsea["library"] == "Reactome_Pathways_2024") & (_gsea["fdr"] < 0.25)]
    _chosen = set()
    for pair in order:
        _block = _sub[_sub["display"] == pair]
        _block = _block.reindex(_block["nes"].abs().sort_values(ascending=False).index)
        _chosen.update(_block.head(20)["pathway"].tolist())
    _chosen = list(_chosen)
    _nes = (_sub[_sub["pathway"].isin(_chosen)]
            .pivot_table(index="pathway", columns="display",
                         values="nes", aggfunc="first")
            .reindex(index=_chosen, columns=order))
    _fdr = (_sub[_sub["pathway"].isin(_chosen)]
            .pivot_table(index="pathway", columns="display",
                         values="fdr", aggfunc="first")
            .reindex(index=_chosen, columns=order))
    _row_order = _nes["Liver - cirrhosis"].fillna(-np.inf).sort_values(ascending=False).index
    _nes = _nes.reindex(_row_order).head(20)
    _fdr = _fdr.reindex(_nes.index)

    def _clean(name, wrap=34):
        n = str(name)
        if " R-HSA-" in n:
            n = n.split(" R-HSA-")[0]
        if len(n) > wrap:
            mid = len(n) // 2
            spaces = [i for i, ch in enumerate(n) if ch == " "]
            if spaces:
                split = min(spaces, key=lambda i: abs(i - mid))
                n = n[:split] + "\n" + n[split + 1:]
        return n

    _data = _nes.values.astype(float)
    _max_abs = float(np.nanmax(np.abs(_data))) if np.isfinite(_data).any() else 1.0
    _vmax = max(1.0, min(3.0, _max_abs))
    _n_rows, _n_cols = _data.shape
    axC.set_xlim(-0.5, _n_cols - 0.5)
    axC.set_ylim(_n_rows - 0.5, -0.5)
    for i in range(_n_rows):
        for j in range(_n_cols):
            if not np.isfinite(_data[i, j]):
                axC.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        facecolor="#EEEEEE", edgecolor="white",
                                        lw=0.5, zorder=1))
    im_c = axC.imshow(_data, cmap=CMAP_NES, vmin=-_vmax, vmax=_vmax,
                      aspect="auto", zorder=2)
    axC.set_xticks(np.arange(-0.5, _n_cols, 1), minor=True)
    axC.set_yticks(np.arange(-0.5, _n_rows, 1), minor=True)
    axC.grid(which="minor", color="white", lw=0.6)
    axC.tick_params(which="minor", length=0)
    for i in range(_n_rows):
        for j in range(_n_cols):
            _f = _fdr.iloc[i, j]
            if pd.notna(_f) and _f < 0.05:
                axC.plot(j, i, marker=".", color="black", markersize=3.5, zorder=4)
    axC.set_xticks(np.arange(_n_cols))
    axC.set_xticklabels([_SHORT_LABEL[c] for c in _nes.columns],
                        rotation=0, ha="center", fontsize=6.5)
    axC.set_yticks(np.arange(_n_rows))
    axC.set_yticklabels([_clean(p) for p in _nes.index], fontsize=6.5)
    axC.tick_params(axis="x", which="major", length=0)
    axC.tick_params(axis="y", which="major", length=0)
    for s in axC.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im_c, ax=axC, fraction=0.025, pad=0.03)
    cb.set_label("NES", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, width=0.6, length=2.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    axC.text(1.0, -0.08,
             "•  FDR < 0.05     (all cells shown pass FDR < 0.25)",
             transform=axC.transAxes, ha="right", va="top",
             fontsize=6.5, color="#333333")
    panel_letter(axC, "C", dx=-0.35, dy=1.01)

    return fig

def build_S4():
    """Build Fig S4 — cross-fold and cross-model stability of PC-derived gene rankings."""
    apply_style()

    import matplotlib.pyplot as plt
    from itertools import combinations

    QUALIFYING = [
        ("Liver", "cirrhosis", CORAL),
        ("Small Intestine - Terminal Ileum", "nodularity", SAGE),
        ("Liver", "steatosis", STEEL),
        ("Lung", "congestion", GOLD),
    ]
    K_JACCARD = [10, 25, 50, 100]
    K_PCS_LIST = [50, 100, 200]
    N_PCS_MAX = 800
    TOP_K_PCS_DEFAULT = 100

    def spaced(tissue: str, cat: str) -> str:
        """Render a pair label with a spaced hyphen, wrapping Small Intestine to 3 lines."""
        if tissue == "Small Intestine - Terminal Ileum":
            return f"Small Intestine -\nTerminal Ileum -\n{cat}"
        return f"{tissue} - {cat}"

    print("[load] loading cache ...")
    X_wb, blood_subjid, _, df_meta_url, df_age = load_cache()
    X_conf = build_confounder_matrix(df_age, blood_subjid)
    X_wb_var, _ = variance_filter(X_wb)
    print(f"[load] {X_wb_var.shape[0]} samples x {X_wb_var.shape[1]:,} genes")

    rf_fi = pd.read_csv(Config.TABLES_DIR / "rf_feature_importances.csv")
    pc_imp_full = pd.read_csv(Config.TABLES_DIR / "pc_gene_importance_full.csv")
    val_df = pd.read_csv(Config.TABLES_DIR / "validation_pc_vs_rf.csv")

    def compute_per_fold_pca(tissue: str, category: str):
        """Return list of per-fold PCA artifacts (loadings, kept PC frame, AUC-ranked PC names)."""
        y, _, n_pos, n_neg = assign_donor_labels(df_meta_url, tissue, category, blood_subjid)
        keep = y.notna()
        y_clean = y[keep].astype(int)
        groups = blood_subjid[keep].astype(str)
        X_sub = X_wb_var.loc[keep]
        X_conf_sub = X_conf.loc[keep]

        cv = StratifiedGroupKFold(n_splits=Config.N_SPLITS, shuffle=True,
                                  random_state=Config.SEED)
        folds = []
        for fold_i, (tr, te) in enumerate(cv.split(X_sub, y_clean, groups=groups), 1):
            Xtr = X_sub.iloc[tr]
            ytr = y_clean.iloc[tr]

            scaler = StandardScaler()
            Xtr_scaled = scaler.fit_transform(Xtr)
            n_comp = min(N_PCS_MAX, Xtr_scaled.shape[0] - 1, Xtr_scaled.shape[1])
            pca = PCA(n_components=n_comp, random_state=Config.SEED)
            Xtr_pcs = pca.fit_transform(Xtr_scaled)

            pc_cols = [f"PC{i+1}" for i in range(n_comp)]
            Xtr_pcs = pd.DataFrame(Xtr_pcs, columns=pc_cols, index=Xtr.index)

            X_covar_tr = X_conf_sub.iloc[tr]
            cov_aucs_pc = _compute_pc_covariate_aucs(Xtr_pcs, X_covar_tr)
            keep_mask_pc = _drop_covariate_aligned_pcs(cov_aucs_pc, Config.COVARIATE_ORTHO_AUC_THRESH)
            Xtr_pcs_filt = Xtr_pcs.loc[:, keep_mask_pc]

            # Compute AUC-based ranking for ALL filtered PCs (once)
            top_pcs_all = _auc_feature_selection(Xtr_pcs_filt, ytr,
                                                  top_k=Xtr_pcs_filt.shape[1])

            folds.append({
                "tr": tr,
                "loadings": pca.components_,
                "Xtr_pcs": Xtr_pcs,
                "top_pcs_all": top_pcs_all,   # ranked list of all filtered PC col-names
                "ytr": ytr,
                "X_conf_tr": X_conf_sub.iloc[tr],
            })
            print(f"    fold {fold_i}: PCA {n_comp} comps, kept {keep_mask_pc.sum()} PCs")
        return folds

    def gene_importance_at_k(fold_art, k_pcs: int) -> np.ndarray:
        """Fit RF at top_k_pcs=k and back-project to a per-gene importance vector."""
        top_pcs = fold_art["top_pcs_all"][:k_pcs]
        selected_idx = [int(c.replace("PC", "")) - 1 for c in top_pcs]

        Xtr_sel = pd.concat([fold_art["Xtr_pcs"][top_pcs], fold_art["X_conf_tr"]], axis=1)

        model = make_rf_model()
        model.fit(Xtr_sel, fold_art["ytr"])

        n_pc_features = len(top_pcs)
        rf_imp_pcs = model.feature_importances_[:n_pc_features].copy()
        if rf_imp_pcs.sum() == 0:
            rf_imp_pcs = np.ones_like(rf_imp_pcs) / len(rf_imp_pcs)
        else:
            rf_imp_pcs = rf_imp_pcs / rf_imp_pcs.sum()

        sel_loadings = fold_art["loadings"][selected_idx, :]
        abs_loadings = np.abs(sel_loadings)
        row_sums = abs_loadings.sum(axis=1, keepdims=True)
        abs_loadings_norm = abs_loadings / row_sums
        gene_imp = (rf_imp_pcs[:, None] * abs_loadings_norm).sum(axis=0)
        return gene_imp

    print("\n[compute] running per-fold PCA + AUC-selection for each qualifying pair")
    per_pair = {}
    for tissue, cat, _color in QUALIFYING:
        print(f"\n  pair: {tissue} | {cat}")
        folds = compute_per_fold_pca(tissue, cat)

        per_fold_imp = np.vstack([gene_importance_at_k(f, TOP_K_PCS_DEFAULT) for f in folds])
        ranks = pd.DataFrame(per_fold_imp.T).rank(ascending=False, method="average").values
        rhos = []
        for i, j in combinations(range(ranks.shape[1]), 2):
            rho = spearmanr(ranks[:, i], ranks[:, j]).correlation
            rhos.append(rho)
        print(f"    cross-fold Spearman rho (K=100): mean {np.mean(rhos):.3f}  "
              f"min {np.min(rhos):.3f}  max {np.max(rhos):.3f}")

        agg_imp_by_k = {}
        for k in K_PCS_LIST:
            if k == TOP_K_PCS_DEFAULT:
                per_fold_k = per_fold_imp
            else:
                per_fold_k = np.vstack([gene_importance_at_k(f, k) for f in folds])
            agg_imp_by_k[k] = per_fold_k.mean(axis=0)

        per_pair[(tissue, cat)] = {
            "gene_index": X_wb_var.columns,
            "per_fold_imp": per_fold_imp,
            "cross_fold_rhos": rhos,
            "agg_imp_by_k": agg_imp_by_k,
        }

    print("\n[compute] panel E — rho(PC-derived, RF) at each K_pcs")
    sens_rho = {}
    for (tissue, cat), info in per_pair.items():
        rho_by_k = {}
        rf_pair = rf_fi[(rf_fi["tissue"] == tissue) & (rf_fi["category"] == cat)]
        rf_series = rf_pair.set_index("gene")["mean_importance"] if not rf_pair.empty else pd.Series(dtype=float)
        for k, imp in info["agg_imp_by_k"].items():
            pc_series = pd.Series(imp, index=info["gene_index"])
            common = pc_series.index.intersection(rf_series.index)
            if len(common) < 10:
                rho_by_k[k] = np.nan
                continue
            rho = spearmanr(pc_series[common], rf_series[common]).correlation
            rho_by_k[k] = rho
        sens_rho[(tissue, cat)] = rho_by_k
        print(f"  {tissue} | {cat}: {rho_by_k}")

    print("\n[compute] panel C — top-K Jaccard overlap (within shared gene universe)")
    jaccard_by_pair = {}
    for (tissue, cat), info in per_pair.items():
        pc_series = pd.Series(info["agg_imp_by_k"][TOP_K_PCS_DEFAULT], index=info["gene_index"])
        rf_pair = rf_fi[(rf_fi["tissue"] == tissue) & (rf_fi["category"] == cat)]
        rf_series = rf_pair.set_index("gene")["mean_importance"] if not rf_pair.empty else pd.Series(dtype=float)

        # Restrict both to the common gene universe (the ~200-400 genes RF ranked).
        # Ranks are then commensurate: top-K of each is drawn from the same pool.
        common = pc_series.index.intersection(rf_series.index)
        pc_common = pc_series.loc[common].sort_values(ascending=False)
        rf_common = rf_series.loc[common].sort_values(ascending=False)

        js = []
        for K in K_JACCARD:
            k = min(K, len(common))
            pc_top = set(pc_common.head(k).index)
            rf_top = set(rf_common.head(k).index)
            inter = len(pc_top & rf_top)
            union = len(pc_top | rf_top)
            js.append(inter / union if union else np.nan)
        jaccard_by_pair[(tissue, cat)] = js
        print(f"  {tissue} | {cat}: common={len(common)}  Jaccard @{K_JACCARD} = "
              f"{[f'{v:.3f}' for v in js]}")

    print("\n[render] building figure S5 (3 panels)")

    fig = plt.figure(figsize=(9.6, 3.0))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.0, 1.0, 1.0],
        wspace=0.42,
        left=0.07, right=0.98, top=0.92, bottom=0.22,
    )
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])

    pair_colors = {(t, c): color for t, c, color in QUALIFYING}
    pair_labels = {(t, c): spaced(t, c) for t, c, _ in QUALIFYING}

    data_A = [per_pair[(t, c)]["cross_fold_rhos"] for t, c, _ in QUALIFYING]
    positions = np.arange(len(QUALIFYING))

    for i, (t, c, color) in enumerate(QUALIFYING):
        rhos = per_pair[(t, c)]["cross_fold_rhos"]
        bp = ax_A.boxplot(
            [rhos], positions=[i], widths=0.55, patch_artist=True,
            showfliers=False, medianprops=dict(color="#333", lw=0.9),
            whiskerprops=dict(color="#333", lw=0.6),
            capprops=dict(color="#333", lw=0.6),
            boxprops=dict(facecolor=color, alpha=0.25, edgecolor=color, lw=0.8),
        )
        # Strip plot (jittered)
        rng = np.random.default_rng(seed=0)
        jitter = rng.uniform(-0.12, 0.12, size=len(rhos))
        ax_A.scatter(np.full(len(rhos), i) + jitter, rhos,
                     s=10, color=color, edgecolor="white", lw=0.4, zorder=3)

    ax_A.set_xticks(positions)
    ax_A.set_xticklabels([pair_labels[(t, c)] for t, c, _ in QUALIFYING],
                         rotation=30, ha="right")
    ax_A.set_ylabel("Cross-fold Spearman rho")
    # Zoom to the meaningful range so the box heights + jitter are visible
    all_A = np.concatenate(data_A)
    lo = float(min(0.90, all_A.min() - 0.02))
    ax_A.set_ylim(lo, 1.0)
    ax_A.axhline(0.95, color=GRAY_LT, lw=0.6, ls=":", zorder=1)
    thin_spines(ax_A)
    panel_letter(ax_A, "A")

    rho_vals = []
    row_labels = []
    row_colors = []
    for t, c, color in QUALIFYING:
        row = val_df[(val_df["tissue"] == t) & (val_df["category"] == c)]
        rho_vals.append(row["spearman_corr"].values[0] if not row.empty else np.nan)
        row_labels.append(pair_labels[(t, c)])
        row_colors.append(color)

    y_B = np.arange(len(row_labels))
    ax_B.barh(y_B, rho_vals, color=row_colors, edgecolor="black", linewidth=0.4)
    ax_B.set_yticks(y_B)
    ax_B.set_yticklabels(row_labels)
    ax_B.invert_yaxis()
    ax_B.set_xlim(0, 0.75)
    ax_B.set_xlabel("Spearman rho (PC vs RF)")
    ax_B.axvline(0.5, color=GRAY_LT, lw=0.6, ls=":", zorder=1)
    for i, v in enumerate(rho_vals):
        ax_B.text(v + 0.02, i, f"{v:.2f}", ha="left", va="center",
                  fontsize=6.5, color="black", fontweight="bold")
    thin_spines(ax_B)
    panel_letter(ax_B, "B")

    for (t, c, color) in QUALIFYING:
        ys = jaccard_by_pair[(t, c)]
        ax_C.plot(K_JACCARD, ys, marker="o", ms=4, lw=1.1,
                  color=color, label=pair_labels[(t, c)])
    ax_C.set_xscale("log")
    ax_C.set_xticks(K_JACCARD)
    ax_C.set_xticklabels([str(k) for k in K_JACCARD])
    ax_C.minorticks_off()
    ax_C.set_xlabel("Top-K genes (shared universe)")
    ax_C.set_ylabel("Top-K overlap (Jaccard)")
    _max_j = max(max(v) for v in jaccard_by_pair.values() if v)
    ax_C.set_ylim(0, max(0.5, _max_j * 1.15))
    ax_C.legend(loc="best", frameon=False, fontsize=5.5,
                labelspacing=0.25, borderpad=0.2)
    thin_spines(ax_C)
    panel_letter(ax_C, "C")

    return fig

def build_S6():
    """Build Fig S6 — MR triage funnels + driver forests + liver corroboration."""
    apply_style()

    import matplotlib.pyplot as plt

    PHENO_LABEL = {
        "NAFLD":            "NAFLD",
        "NASH":             "NASH",
        "CIRRHOSIS_BROAD":  "Broad Cirrhosis",
        "K11_FIBROCHIRLIV": "Fibrosis / Cirrhosis",
        "CHIRHEP_NAS":      "Chronic Hepatitis",
        "FIBROLIV":         "Liver Fibrosis",
    }

    T = Config.TABLES_DIR
    mr_st_drv = pd.read_csv(T / "mr_blood_steatosis_up100_same_direction.csv")
    mr_cr_drv = pd.read_csv(T / "mr_blood_cirrhosis_up100_same_direction.csv")
    mr_st_prt = pd.read_csv(T / "mr_blood_steat_down_opposite_direction.csv")
    mr_cr_prt = pd.read_csv(T / "mr_blood_cirr_down_opposite_direction.csv")

    # Top 100 blood FC tables (up direction) — the funnel numerator source.
    b_st_up = pd.read_csv(T / "blood_fc_steatosis_top100_up.csv")
    b_cr_up = pd.read_csv(T / "blood_fc_cirrhosis_top100_up.csv")

    def funnel_counts(mr_drv, mr_prt, n_top=100):
        """Derive MR funnel stage counts from the driver/protective tables."""
        olink = mr_drv["Gene"].nunique()
        outcome = mr_drv[mr_drv["p_out"] < 0.05]["Gene"].nunique()
        drivers = outcome
        prot = mr_prt[mr_prt["p_out"] < 0.05]["Gene"].nunique()
        return {"top": n_top, "olink": olink, "outcome": outcome,
                "drivers": drivers, "prot": prot}

    CNT_ST = funnel_counts(mr_st_drv, mr_st_prt)
    CNT_CR = funnel_counts(mr_cr_drv, mr_cr_prt)

    COL_STEAT   = "#E19137"
    COL_CIRR    = STEEL
    COL_OTHER   = SLATE
    FUNNEL_FILL = ["#FBE9D6", "#F4CCA1", "#E19137", "#B96A18"]
    FUNNEL_CIRR = ["#DDE5F0", "#B0C3DE", STEEL,   "#2E4A6A"]

    def draw_funnel(ax, counts, colors, pair_label, prot_n):
        """Vertical stack of 4 boxes with counts inside, arrows between."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        stages = [
            ("Top 100 blood FC (up)",       counts["top"]),
            ("Olink + cis-pQTL",            counts["olink"]),
            ("Outcome significant (p < 0.05)", counts["outcome"]),
            ("Driver candidates",           counts["drivers"]),
        ]
        n = len(stages)
        # box widths shrink as we descend (funnel visual)
        widths = np.linspace(0.98, 0.55, n)
        heights = 0.14
        top = 0.98
        gap = 0.03
        y_positions = []
        for i, ((lbl, val), w, col) in enumerate(zip(stages, widths, colors)):
            y_top = top - i * (heights + gap)
            y_bot = y_top - heights
            x0 = 0.5 - w / 2
            y_positions.append((y_top, y_bot))
            rect = FancyBboxPatch(
                (x0, y_bot), w, heights,
                boxstyle="round,pad=0.005,rounding_size=0.015",
                linewidth=0.6, edgecolor="#333333", facecolor=col, zorder=2)
            ax.add_patch(rect)
            ax.text(0.5, y_bot + heights * 0.62, lbl,
                    ha="center", va="center", fontsize=7,
                    color="black", zorder=3)
            ax.text(0.5, y_bot + heights * 0.25, f"n = {val}",
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color="black", zorder=3)

        for i in range(n - 1):
            y_from = y_positions[i][1]
            y_to   = y_positions[i + 1][0]
            arr = FancyArrowPatch(
                (0.5, y_from), (0.5, y_to),
                arrowstyle="-|>", mutation_scale=8,
                linewidth=0.9, color="#333333", zorder=1)
            ax.add_patch(arr)

        y_bot_final = y_positions[-1][1]
        ax.text(0.5, y_bot_final - 0.03,
                f"Protective (down 100 → sig): n = {prot_n}",
                ha="center", va="top", fontsize=7.5,
                fontweight="bold", color=COL_OTHER)

    def prepare_forest(mr_df, filter_col="p_out", thresh=0.05, hi_gene=None,
                       hi_color=None, base_color=SLATE, top_k=None):
        """Return (rows, y_labels, colors) sorted by ascending p_MR, one row per gene."""
        d = mr_df[mr_df[filter_col] < thresh].copy()
        # For each gene, keep the row with smallest p_MR
        d = d.sort_values("p_MR").drop_duplicates("Gene", keep="first")
        d = d.sort_values("p_MR", ascending=True)
        if top_k is not None:
            d = d.head(top_k)
        # Reverse so smallest p is at top of the plot
        d = d.iloc[::-1].reset_index(drop=True)
        colors = [hi_color if (hi_gene is not None and g == hi_gene) else base_color
                  for g in d["Gene"]]
        y_labels = [
            f"{g}   |   {PHENO_LABEL.get(p, p)}"
            for g, p in zip(d["Gene"], d["Phenotype"])
        ]
        return d, y_labels, colors

    def draw_forest(ax, d, y_labels, colors, xlabel):
        n = len(d)
        y = np.arange(n)
        for yi, (_, row) in enumerate(d.iterrows()):
            b, se = row["beta_out"], row["SE"]
            ax.plot([b - 1.96 * se, b + 1.96 * se], [yi, yi],
                    color="0.45", lw=0.7, zorder=2)
            ax.plot(b, yi, "o", color=colors[yi], markersize=5,
                    markeredgecolor="black", markeredgewidth=0.35, zorder=3)
        ax.axvline(0, color="0.4", ls="--", lw=0.6, alpha=0.7, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels, fontsize=6)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylim(-0.7, n - 0.3)
        thin_spines(ax)

    def load_blood_vs_liver(pair: str):
        """Return blood-vs-liver-tissue log2 FC for the top-100 blood-up genes."""
        b = pd.read_csv(T / f"blood_fc_{pair}_top100_up.csv")
        l = pd.read_csv(T / f"liver_tissue_fc_{pair}_olink.csv")
        m = b[["gene_id", "gene_name", "log2fc"]].merge(
            l[["gene_id", "log2fc"]], on="gene_id", how="inner",
            suffixes=("_blood", "_liver"))
        return m

    scat_cirr = load_blood_vs_liver("cirrhosis")
    scat_steat = load_blood_vs_liver("steatosis")

    # Kept for callers that still want a single fig; panels A/B/C are also
    # available individually via build_S6A/B/C for the PDF-composition path.
    fig_a = build_S6A(mr_st_drv, mr_st_prt)
    fig_b = build_S6B(mr_st_drv)
    fig_c = build_S6C(mr_st_prt)
    plt.close(fig_a); plt.close(fig_b); plt.close(fig_c)
    # Fall through — return a compact single-panel figure for the old inline
    # embed path. NB17 now uses build_S6A/B/C + compose_vector.compose_figure_S6.
    return build_S6A(mr_st_drv, mr_st_prt)

def build_S6A(mr_st_drv, mr_st_prt):
    """Fig S6 Panel A — Liver-steatosis MR triage funnel (Fig 4A design)."""
    from trace_path.main_figure_panels import draw_mr_funnel, FUNNEL_STEAT
    fig = plt.figure(figsize=(5.28, 5.11))
    ax = fig.add_subplot(1, 1, 1)
    draw_mr_funnel(fig, ax,
                   mr_drv=mr_st_drv, mr_prt=mr_st_prt,
                   palette=FUNNEL_STEAT,
                   title="Liver - Steatosis MR triage")
    return fig

def build_S6B(mr_st_drv):
    """Fig S6 Panel B — Steatosis Causal Drivers strip (Fig 4F design)."""
    from trace_path.main_figure_panels import (
        draw_causal_panel, order_rows_auto, _COL_DRV as DRV_ORANGE,
    )
    fig = plt.figure(figsize=(9.0, 8.5))
    gs = GridSpec(1, 1, figure=fig,
                  left=0.05, right=0.98, top=0.82, bottom=0.12)
    drv_st = order_rows_auto(mr_st_drv, sig_col="p_MR", sig_thresh=0.05,
                             gene_sort_col="p_MR")
    draw_causal_panel(
        fig, gs[0], drv_st, DRV_ORANGE,
        title="Liver - Steatosis Causal Drivers",
        subtitle="upregulated genes + same-direction MR",
        fibro_label="FIBROLIVER",
    )
    return fig

def build_S6C(mr_st_prt):
    """Fig S6 Panel C — Steatosis Causal Protective strip (Fig 4G design)."""
    from trace_path.main_figure_panels import (
        draw_causal_panel, order_rows_auto, _COL_PRT as DRV_PURPLE,
    )
    fig = plt.figure(figsize=(9.0, 8.5))
    gs = GridSpec(1, 1, figure=fig,
                  left=0.05, right=0.98, top=0.82, bottom=0.12)
    prt_st = order_rows_auto(mr_st_prt, sig_col="p_MR", sig_thresh=0.05,
                             gene_sort_col="p_MR")
    # Wider strip column so the wrapped `K11\nFIBROCHIRLIV` label doesn't overlap the SERPINA11 gene row.
    draw_causal_panel(
        fig, gs[0], prt_st, DRV_PURPLE,
        title="Liver - Steatosis Causal Protective",
        subtitle="downregulated genes + opposite-direction MR",
        fibro_label="FIBROLIVER",
        strip_width_ratio=1.15,
    )
    return fig

