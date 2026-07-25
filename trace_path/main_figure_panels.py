"""Panel builders for main figures 2, 3, 4."""
from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.stats import mannwhitneyu

from trace_path.config import Config
from trace_path.figure_builder import (
    apply_house_style, hs_axes, hs_panel_letter, hs_dumbbell,
    PRIMARY_BLUE, DRIVER_YELLOW, PROTECTIVE_PURPLE, HIGHLIGHT_RED, NEUTRAL_GREY, BAND_TAN,
    INK, MUTED_GREY, CMAP_HOUSE_AUC,
    pretty_pair, tissue_display, FINNGEN_LABEL, EN_DASH,
    CMAP_AUC, CMAP_NES,
    thin_spines, CORAL, STEEL,
)

TBL = Config.TABLES_DIR

def load_fig2_data():
    """CSV-only inputs for Figure 2 panels A + B."""
    return {
        "pair_bal":     pd.read_csv(TBL / "pair_class_balance.csv"),
        "donor_scores": pd.read_csv(TBL / "pca_donor_scores_wb20k.csv", index_col=0),
        "pca_var":      pd.read_csv(TBL / "pca_explained_variance_top10.csv"),
    }

def compute_fig2C_covariate_auc_matrix(n_pcs=50):
    """Recompute the per-PC covariate-AUC matrix used by Figure 2C."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedGroupKFold
    from trace_path.data import variance_filter, build_confounder_matrix
    from trace_path.labels import assign_donor_labels
    from trace_path.models import (
        _compute_pc_covariate_aucs, _drop_covariate_aligned_pcs,
    )

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
    cv = StratifiedGroupKFold(n_splits=Config.N_SPLITS, shuffle=True,
                              random_state=Config.SEED)
    splits = list(cv.split(X_sub.values, y_clean.values, groups=g.values))
    tr_idx, _ = splits[0]
    Xtr = X_sub.iloc[tr_idx]
    Xtr_scaled = StandardScaler().fit_transform(Xtr)
    n_comp = min(800, Xtr_scaled.shape[0] - 1, Xtr_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=Config.SEED).fit(Xtr_scaled)
    scores_tr = pca.transform(Xtr_scaled)
    pcs_df = pd.DataFrame(scores_tr,
                          columns=[f"PC{i+1}" for i in range(n_comp)],
                          index=Xtr.index)
    aucs = _compute_pc_covariate_aucs(pcs_df, X_conf_sub.iloc[tr_idx])
    keep_mask = _drop_covariate_aligned_pcs(
        aucs, Config.COVARIATE_ORTHO_AUC_THRESH
    )
    return aucs.head(n_pcs), np.asarray(keep_mask[:n_pcs], dtype=bool)

BAR_BLUE   = "#087FB8"
BAR_GOLD   = "#F2A000"
DASH_GREY  = "#8A8A8A"
BRANCH_GREY = "#5F5F5F"
BAND_GREY  = "#F3F3F3"

COV_SPECS_2B = [
    ("TRISCHD", "Ischemic time", "viridis", False),
    ("AGE",     "Age",           "viridis", False),
    ("SEX",     "Sex",           None,      True),
    ("RACE",    "Race",          None,      True),
    ("DTHHRDY", "Hardy scale",   "plasma",  False),
]

# GTEx v10 phenotype code → human-readable label (dbGaP data dictionary).
SEX_LABELS_2B = {1: "Male", 2: "Female"}
RACE_LABELS_2B = {1: "Asian", 2: "Black", 3: "White", 4: "AI/AN"}


def panel_fig2B(fig, gs_slot, donor_scores, pca_var):
    """Panel B: 5-panel PC1 x PC2 covariate scatters (Jul-10 design)."""
    pc1_var = pca_var.iloc[0]["var_ratio"] * 100
    pc2_var = pca_var.iloc[1]["var_ratio"] * 100
    inner = gs_slot.subgridspec(1, 5, wspace=0.15)
    axes = []
    ax0 = None
    for i, (col, label, cmap, is_cat) in enumerate(COV_SPECS_2B):
        ax = fig.add_subplot(inner[0, i], sharex=ax0, sharey=ax0)
        if is_cat:
            label_map = SEX_LABELS_2B if col == "SEX" else RACE_LABELS_2B
            for v in sorted(donor_scores[col].dropna().unique()):
                sub = donor_scores[donor_scores[col] == v]
                leg = label_map.get(int(v), f"{v:g}")
                ax.scatter(sub["PC1"], sub["PC2"], s=8, alpha=0.7, label=leg)
            ax.legend(fontsize=7, frameon=False, loc="best",
                      handletextpad=0.3, borderaxespad=0.3)
        else:
            sc = ax.scatter(donor_scores["PC1"], donor_scores["PC2"],
                            s=8, alpha=0.75, c=donor_scores[col], cmap=cmap)
            cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
            cb.solids.set_rasterized(False)
            cb.solids.set_edgecolor("face")
            if col == "DTHHRDY":
                cb.set_ticks([0, 1, 2, 3, 4])
                cb.set_ticklabels(["0", "1", "2", "3", "4"])
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.3, lw=0.4)
        thin_spines(ax)
        if i == 0:
            ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)", fontsize=9)
            ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)", fontsize=9)
            ax0 = ax
        else:
            ax.tick_params(axis="y", labelleft=False)
        axes.append(ax)
    return axes

COV_ORDER_2C  = ["TRISCHD", "AGE", "SEX", "RACE", "DTHHRDY"]
COV_LABELS_2C = ["Ischemic time", "Age", "Sex", "Race", "Hardy scale"]

def panel_fig2C(fig, ax, cov_aucs, pc_keep_mask):
    """Panel C: per-PC covariate AUC heatmap — approved landscape design."""
    dropped_pc_idx = np.flatnonzero(~pc_keep_mask)
    mat = cov_aucs[COV_ORDER_2C].values.T  # (5, N_PCs)
    n_r, n_c = mat.shape
    X = np.arange(n_c + 1) - 0.5
    Y = np.arange(n_r + 1) - 0.5
    im = ax.pcolormesh(X, Y, mat, cmap=CMAP_AUC,
                       vmin=0.5, vmax=1.0, shading="flat",
                       edgecolors="none", antialiased=True)
    im.set_rasterized(False)
    ax.set_aspect("auto")
    ax.set_ylim(n_r - 0.5, -0.5)   # keep imshow-like top-to-bottom ordering
    ax.set_yticks(range(len(COV_LABELS_2C)))
    ax.set_yticklabels(COV_LABELS_2C, fontsize=7)
    xt = [0, 9, 19, 29, 39, 49]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"PC{i+1}" for i in xt], fontsize=6.5, rotation=0)
    ax.set_xlabel("Principal component", fontsize=8)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > Config.COVARIATE_ORTHO_AUC_THRESH:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=CORAL, lw=1.0, zorder=4))

    if len(dropped_pc_idx):
        ax.scatter(
            dropped_pc_idx, np.full(len(dropped_pc_idx), -0.72),
            marker="v", s=20, color=CORAL, edgecolor="white", linewidth=0.35,
            clip_on=False, zorder=6,
        )

    thr = Config.COVARIATE_ORTHO_AUC_THRESH
    ax.text(
        1.0, 1.05,
        f"Coral outline: AUC > {thr:.2f}   ▼ PC dropped",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color=INK, clip_on=False,
    )

    cax = ax.inset_axes([1.02, 0.0, 0.02, 1.0])
    cb = fig.colorbar(im, cax=cax)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")
    cb.set_label("AUC", fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=1.5, width=0.4)
    thin_spines(ax)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=1.5, width=0.4)


def load_fig3_data():
    return {
        "pc_univar":    pd.read_csv(TBL / "liver_cirrhosis_pc_univariate_auc.csv"),
        "pc_rf_gini":   pd.read_csv(TBL / "liver_cirrhosis_pc_rf_gini.csv"),
        "six_mat_mean": pd.read_csv(TBL / "six_model_auc_matrix_mean.csv"),
        "pc_results":   pd.read_csv(TBL / "pc_auc_results.csv"),
        "gene_imp":     pd.read_csv(TBL / "pc_gene_importance_full.csv"),
        "gsea":         pd.read_csv(TBL / "gsea_pathway_enrichment.csv"),
    }

QUAL_ORDER_F3 = [
    ("Liver", "cirrhosis"),
    ("Small Intestine - Terminal Ileum", "nodularity"),
    ("Lung", "congestion"),
    ("Liver", "steatosis"),
]

def panel_fig3A(fig, ax, pc_univar):
    """Top-10 single-feature PC AUCs (Liver – Cirrhosis) — Jul-15 approved design."""
    top = pc_univar.sort_values("mean_auc", ascending=False).head(10).iloc[::-1]
    y = np.arange(len(top))
    ax.set_axisbelow(True)  # send grid + reference line below bars/errorbars
    # Grid drawn per-tick, skipping 0.6 so it doesn't collide with the
    # dash-dot reference line drawn at 0.6.
    for xt in (0.5, 0.7, 0.8):
        ax.axvline(xt, color="#EEEEEE", linestyle="--", linewidth=0.4,
                   alpha=0.9)
    # Thicker dash-dot-dash reference line at 0.6 (AUC gate)
    ax.axvline(0.6, color="#555555", linestyle=(0, (6, 3, 1, 3)),
               linewidth=1.0, alpha=0.9)
    ax.barh(y, top["mean_auc"], xerr=top["std_auc"], height=0.62,
            color=PRIMARY_BLUE, edgecolor="none", zorder=3,
            error_kw={"elinewidth": 0.6, "capsize": 3, "capthick": 0.6,
                      "ecolor": INK, "zorder": 4})
    ax.set_yticks(y)
    ax.set_yticklabels([f"PC{int(pc)}" for pc in top["pc"]], fontsize=9)
    ax.set_xlabel("Single-feature AUC", fontsize=10)
    ax.set_xlim(0.43, 0.83)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    ax.tick_params(axis="x", labelsize=9)
    hs_axes(ax, spines=("left", "bottom"))

def panel_fig3B(fig, ax, six_mat_mean):
    """4x4 model x qualifying-pair AUC heatmap —  R design."""
    # Hyphen-only labels (no en-dashes); wrap Small Intestine onto 3 lines
    # (no hyphens on the breaks — line breaks alone act as the separator)
    def _hyphen_label(t, c):
        tt = str(t)
        cc = str(c)
        cc = cc[:1].upper() + cc[1:] if cc else cc
        if "Small Intestine" in tt:
            return f"Small Intestine\nTerminal Ileum\n{cc}"
        return f"{tt} - {cc}"
    qual_labels = [_hyphen_label(t, c) for t, c in QUAL_ORDER_F3]
    qual_raw = [f"{t} - {c}" for t, c in QUAL_ORDER_F3]

    models_4 = ["LR raw", "RF raw", "PC-LR (expr+covars)", "PC-RF (expr+covars)"]
    # Two-line model labels (approved Jul-15 style)
    model_labels = ["Logistic\nRegression", "Random\nForest",
                    "Logistic\nRegression", "Random\nForest"]

    pairs_df = six_mat_mean.set_index("pathology").reindex(qual_raw)
    # Order rows by rowMeans (best-performing pathology at top) —  R
    row_means = pairs_df[models_4].mean(axis=1)
    order_idx = row_means.sort_values(ascending=False).index.tolist()
    pairs_df = pairs_df.reindex(order_idx)
    raw_to_label = dict(zip(qual_raw, qual_labels))
    sorted_labels = [raw_to_label[r] for r in order_idx]
    mat = pairs_df[models_4].values

    # 3-colour ramp: BLUE -> orange -> RED
    house_ramp = LinearSegmentedColormap.from_list(
        "house_3B", ["#5B7BA6", "#E8A94E", "#C0554E"], N=256)
    vmin = 0.5
    vmax = float(np.nanmax(mat))
    n_r, n_c = mat.shape
    X = np.arange(n_c + 1) - 0.5
    Y = np.arange(n_r + 1) - 0.5
    im = ax.pcolormesh(X, Y, mat, cmap=house_ramp,
                       vmin=vmin, vmax=vmax, shading="flat",
                       edgecolors="none", antialiased=True, zorder=1)
    im.set_rasterized(False)
    ax.set_aspect("auto")
    ax.set_ylim(n_r - 0.5, -0.5)   # keep imshow-like top-to-bottom ordering

    # Dashed white separators between cells (matches approved screenshot)
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4,
            linestyle=(0, (4, 3)))
    ax.tick_params(which="minor", length=0)

    ax.set_xticks(range(4))
    ax.set_xticklabels(model_labels, fontsize=9, ha="center",
                       linespacing=1.15)
    ax.set_yticks(range(4))
    ax.set_yticklabels(sorted_labels, fontsize=9, linespacing=1.15)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=INK, zorder=3)

    # Winning-column outline: PC-RF (expr+covars) = col 3, long dashes
    ax.add_patch(Rectangle((3 - 0.5, -0.5), 1, 4, fill=False,
                           edgecolor=INK, linewidth=1.8,
                           linestyle=(0, (10, 4)), zorder=5))

    # Two-tier group headers UNDER the model labels with connecting horizontal
    # SOLID lines (small gap in the middle differentiates the two groups).
    # GROUP_HLINE_Y controls how close the lines sit to the model labels
    # (less negative = higher = closer to model names). Adjust to taste.
    GROUP_HLINE_Y = -0.08
    GROUP_TEXT_Y  = GROUP_HLINE_Y - 0.02
    ax.text(0.5, GROUP_TEXT_Y, "Raw Expression", ha="center", va="top",
            fontsize=9, color=INK, transform=ax.get_xaxis_transform())
    ax.text(2.5, GROUP_TEXT_Y, "Expression + Covariates", ha="center", va="top",
            fontsize=9, color=INK, transform=ax.get_xaxis_transform())
    ax.hlines(y=GROUP_HLINE_Y, xmin=-0.5, xmax=1.4,
              transform=ax.get_xaxis_transform(),
              colors=INK, linewidth=0.9, linestyles="-", clip_on=False)
    ax.hlines(y=GROUP_HLINE_Y, xmin=1.6, xmax=3.5,
              transform=ax.get_xaxis_transform(),
              colors=INK, linewidth=0.9, linestyles="-", clip_on=False)

    cax = ax.inset_axes([1.03, 0.0, 0.03, 1.0])
    cb = fig.colorbar(im, cax=cax)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")
    cb.set_label("Mean CV AUC", fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=1.5, width=0.4)

    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="both", length=0)

def panel_fig3C(fig, ax, pc_results):
    """Dumbbell: covariates-only vs PC+covariates —  R design."""
    COVAR_TAN = "#C7A96B"
    CONN_TAN = "#DEC998"  # lighter tan connector

    c_order = [
        ("Liver", "cirrhosis"),
        ("Lung", "congestion"),
        ("Liver", "steatosis"),
        ("Small Intestine - Terminal Ileum", "nodularity"),
    ]

    def _hyphen_label(t, c):
        cc = c[:1].upper() + c[1:]
        if "Small Intestine" in t:
            return f"Small Intestine\nTerminal Ileum\n{cc}"
        return f"{t} - {cc}"

    rows = []
    for t, c in c_order:
        r = pc_results[(pc_results["tissue"] == t) & (pc_results["category"] == c)]
        if not r.empty:
            rows.append({
                "label": _hyphen_label(t, c),
                "auc_conf": r["auc_conf"].iloc[0],
                "auc_comb": r["auc_pc_conf"].iloc[0],
            })
    dumb = pd.DataFrame(rows)
    dumb["delta"] = dumb["auc_comb"] - dumb["auc_conf"]

    ax.set_axisbelow(True)
    # Muted grid drawn per-tick, skipping 0.6 so it doesn't collide with the
    # dash-dot reference line at 0.6 (grid + reference at the same x muddies
    # the pattern — Panel 3A escapes this only because it uses fewer ticks).
    for xt in (0.5, 0.7, 0.8, 0.9):
        ax.axvline(xt, color="#EEEEEE", linestyle="--", linewidth=0.4,
                   alpha=0.9, zorder=0)
    # Thicker dash-dot-dash reference line at 0.6 (AUC gate) — matches Panel 3A
    ax.axvline(0.6, color="#555555", linestyle=(0, (6, 3, 1, 3)),
               linewidth=1.0, alpha=0.9, zorder=0)

    y = np.arange(len(dumb))
    for i, r in dumb.iterrows():
        hs_dumbbell(ax, y=i, x_left=r.auc_conf, x_right=r.auc_comb,
                      color_left=COVAR_TAN, bar_color=CONN_TAN,
                      labels_delta=False, zorder_bar=2)
        ax.text(r.auc_comb + 0.012, i, f"Δ = {r.delta:.2f}",
                ha="left", va="center", fontsize=8.5,
                fontweight="bold", color=HIGHLIGHT_RED, zorder=6)

    ax.set_yticks(y)
    ax.set_yticklabels(dumb["label"], fontsize=9, linespacing=1.15)
    ax.invert_yaxis()
    ax.set_xlim(0.43, 0.92)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("Mean AUC", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    leg_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COVAR_TAN,
               markeredgecolor="white", markeredgewidth=0.6, markersize=9,
               label="Covariates only"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PRIMARY_BLUE,
               markeredgecolor="white", markeredgewidth=0.6, markersize=9,
               label="PC + covariates"),
    ]
    ax.legend(handles=leg_handles, loc="lower right", frameon=False,
              fontsize=9, handletextpad=0.4, borderaxespad=0.4)
    hs_axes(ax, spines=("left", "bottom"))

def panel_fig3D(fig, ax, pc_rf_gini):
    """RF Gini importance for top-10 PCs (Liver - Cirrhosis)."""
    top_gini = pc_rf_gini.sort_values("mean_gini", ascending=False).head(10).iloc[::-1]
    y = np.arange(len(top_gini))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#EEEEEE", linestyle="--", linewidth=0.4,
            alpha=0.9)
    ax.barh(y, top_gini["mean_gini"], xerr=top_gini["std_gini"], height=0.62,
            color=PRIMARY_BLUE, edgecolor="none", zorder=3,
            error_kw={"elinewidth": 0.6, "capsize": 3, "capthick": 0.6,
                      "ecolor": INK, "zorder": 4})
    ax.set_yticks(y)
    ax.set_yticklabels([f"PC{int(pc)}" for pc in top_gini["pc"]], fontsize=9)
    ax.set_xlabel("RF Gini importance (Liver - Cirrhosis)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    hs_axes(ax, spines=("left", "bottom"))

def panel_fig3E(fig, ax, gene_imp):
    """Top-10 back-projected named-gene importance (Liver - Cirrhosis)."""
    gi = gene_imp[(gene_imp["tissue"] == "Liver") & (gene_imp["category"] == "cirrhosis")]
    top_g = gi.sort_values("importance", ascending=False).head(10).iloc[::-1]
    y = np.arange(len(top_g))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#EEEEEE", linestyle="--", linewidth=0.4,
            alpha=0.9)
    ax.barh(y, top_g["importance"], height=0.62,
            color=PRIMARY_BLUE, edgecolor="none", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(top_g["gene_name"], fontsize=9)
    ax.set_xlabel("Back-projected gene importance (Liver - Cirrhosis)",
                  fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    hs_axes(ax, spines=("left", "bottom"))

LABEL_MAP_F3F = {
    "Liver | cirrhosis":                              "Liver - cirrhosis",
    "Liver | steatosis":                              "Liver - steatosis",
    "Lung | congestion":                              "Lung - congestion",
    "Small Intestine - Terminal Ileum | nodularity":  "Small Intestine - Terminal Ileum - nodularity",
}
PAIR_ORDER_F3F = ["Liver - cirrhosis", "Liver - steatosis",
                  "Lung - congestion", "Small Intestine - Terminal Ileum - nodularity"]
SHORT_LABEL_F3F = {
    "Liver - cirrhosis": "Liver -\ncirrhosis",
    "Liver - steatosis": "Liver -\nsteatosis",
    "Lung - congestion": "Lung -\ncongestion",
    "Small Intestine - Terminal Ileum - nodularity":
        "Small Intestine -\nTerminal Ileum -\nnodularity",
}

KEGG_LABEL_OVERRIDES = {
    # KEGG canonical name -> shorter display form for the heatmap.
    "VIRAL PROTEIN INTERACTION WITH CYTOKINE AND CYTOKINE RECEPTOR":
        "VIRAL PROTEIN–CYTOKINE RECEPTOR INTERACTION",
}

def _kegg_upper_label(name):
    """Uppercase pathway label, strip trailing 'R-HSA-...' suffix. No wrapping."""
    n = str(name)
    if " R-HSA-" in n:
        n = n.split(" R-HSA-")[0]
    n = n.upper()
    return KEGG_LABEL_OVERRIDES.get(n, n)

def panel_fig3F(fig, ax, gsea, k_top=20):
    """KEGG_2026 top-20 union NES heatmap across 4 qualifying pairs."""
    g = gsea.copy()
    g["display"] = g["label"].map(LABEL_MAP_F3F)
    sub = g[(g["library"] == "KEGG_2026") & (g["fdr"] < 0.25)]
    chosen = set()
    for pair in PAIR_ORDER_F3F:
        block = sub[sub["display"] == pair]
        block = block.reindex(block["nes"].abs().sort_values(ascending=False).index)
        chosen.update(block.head(k_top)["pathway"].tolist())
    chosen = list(chosen)
    nes = (sub[sub["pathway"].isin(chosen)]
           .pivot_table(index="pathway", columns="display",
                        values="nes", aggfunc="first")
           .reindex(index=chosen, columns=PAIR_ORDER_F3F))
    fdr = (sub[sub["pathway"].isin(chosen)]
           .pivot_table(index="pathway", columns="display",
                        values="fdr", aggfunc="first")
           .reindex(index=chosen, columns=PAIR_ORDER_F3F))
    row_order = nes["Liver - cirrhosis"].fillna(-np.inf).sort_values(ascending=False).index
    nes = nes.reindex(row_order).head(k_top)
    fdr = fdr.reindex(nes.index)

    data = nes.values.astype(float)
    max_abs = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 1.0
    vmax = max(1.0, min(3.0, max_abs))
    n_rows, n_cols = data.shape
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    # Grey cells for missing/non-enriched
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isfinite(data[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor="#EEEEEE", edgecolor="white",
                                       lw=0.5, zorder=1))
    # pcolormesh keeps cells as vector quads in PDF (imshow rasterizes).
    # No per-cell edges here — thin minor-grid separators (below) match the
    # earlier imshow look while preserving vector output.
    X = np.arange(n_cols + 1) - 0.5
    Y = np.arange(n_rows + 1) - 0.5
    im = ax.pcolormesh(X, Y, np.ma.masked_invalid(data),
                       cmap=CMAP_NES, vmin=-vmax, vmax=vmax,
                       shading="flat", edgecolors="none",
                       zorder=2, antialiased=True)
    ax.set_aspect("auto")
    # Thin white separators between cells (minor grid) — matches imshow-era look
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6, zorder=3)
    ax.tick_params(which="minor", length=0)
    # Solid black dot for FDR < 0.05
    for i in range(n_rows):
        for j in range(n_cols):
            f = fdr.iloc[i, j]
            if pd.notna(f) and f < 0.05:
                ax.plot(j, i, marker="o", markerfacecolor=INK,
                        markeredgecolor="white", markeredgewidth=0.4,
                        markersize=4.5, zorder=5)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([SHORT_LABEL_F3F[c] for c in nes.columns],
                       rotation=0, ha="center", fontsize=7)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([_kegg_upper_label(p) for p in nes.index], fontsize=6.5)
    ax.tick_params(axis="both", which="major", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # NES colorbar on right — keep vector
    im.set_rasterized(False)
    cb_ax = ax.inset_axes([1.03, 0.0, 0.03, 1.0])
    cb = fig.colorbar(im, cax=cb_ax)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")
    cb.set_label("NES", fontsize=8)
    cb.ax.tick_params(labelsize=7, width=0.6, length=2.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    ax.text(
        0.0, -0.14,
        "•  FDR < 0.05     (all cells shown pass FDR < 0.25)"
        "     grey cell: pathway not enriched in that pair",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=6.5, color=INK, clip_on=False,
    )


PHEN_ORDER_SEVERITY = ["K11_FIBROCHIRLIV", "FIBROLIV", "CIRRHOSIS_BROAD",
                       "CHIRHEP_NAS", "NASH", "NAFLD"]
PHEN_LABEL_ROW1D = {
    "NAFLD": "NAFLD", "NASH": "NASH",
    "CIRRHOSIS_BROAD": "Broad\nCirrhosis",
    "K11_FIBROCHIRLIV": "Fibrosis /\nCirrhosis",
    "CHIRHEP_NAS": "Chronic\nHepatitis",
    "FIBROLIV": "Liver\nFibrosis",
}
FUNNEL_CIRR = ["#DDE5F0", "#B0C3DE", PRIMARY_BLUE, "#2E4A6A"]

def load_fig4_data(gene="SERPINE1"):
    T = Config.TABLES_DIR
    mr_cr_drv = pd.read_csv(T / "mr_blood_cirrhosis_up100_same_direction.csv")
    mr_cr_prt = pd.read_csv(T / "mr_blood_cirr_down_opposite_direction.csv")
    pc_auc    = pd.read_csv(T / "pc_auc_results.csv")

    st7_file = Config.RAW_DIR / "41586_2023_6592_MOESM3_ESM.xlsx"
    st9 = pd.read_excel(st7_file, sheet_name="ST9", header=4)
    st9["gene_symbol"] = st9["UKBPPP ProteinID"].astype(str).str.split(":").str[0]
    se_exp = (st9[st9["cis/trans"] == "cis"]
              .drop_duplicates("gene_symbol")
              .set_index("gene_symbol")["SE (discovery)"].to_dict())

    blood_tpm = pd.read_parquet(Config.CACHE_DIR / "blood_tpm.parquet")
    b_samp_cols = [c for c in blood_tpm.columns if c.startswith("GTEX-")]
    b_gene_col = "gene_name" if "gene_name" in blood_tpm.columns else "Description"
    blood_vals = (blood_tpm.loc[blood_tpm[b_gene_col] == gene, b_samp_cols]
                  .iloc[0].astype(float))

    lab = pd.read_csv(Config.PROCESSED_DIR / "liver_pathology_labels_imputed.csv")
    lab["has_cirrhosis"] = (lab["Pathology.Categories.Final"].fillna("")
                            .str.contains("cirrhosis", case=False, na=False)
                            .astype(int))
    donor_cirr = lab.groupby("Subject.ID")["has_cirrhosis"].max().to_dict()

    def _subj(sampid):
        p = sampid.split("-"); return f"{p[0]}-{p[1]}"

    b_df = pd.DataFrame({"sampid": b_samp_cols,
                         "subj":   [_subj(s) for s in b_samp_cols],
                         "tpm":    blood_vals.values})
    b_df["cirr"] = b_df["subj"].map(donor_cirr)
    b_df = b_df.dropna(subset=["cirr"]).astype({"cirr": int})
    b_pos = b_df.loc[b_df.cirr == 1, "tpm"].values
    b_neg = b_df.loc[b_df.cirr == 0, "tpm"].values
    b_stat, b_p = mannwhitneyu(b_pos, b_neg, alternative="two-sided")
    b_log2fc = np.log2((b_pos.mean() + 1) / (b_neg.mean() + 1))

    mr_serpine = mr_cr_drv[mr_cr_drv.Gene == gene].copy()
    se_serpine = se_exp.get(gene, np.nan)
    beta_exp_serpine = float(mr_serpine["beta_exp"].iloc[0])

    val = pd.read_csv(T / "external_validation_liver_cirrhosis_GSE142255.csv")
    gtex_auc = float(
        pc_auc.loc[(pc_auc["tissue"] == "Liver") & (pc_auc["category"] == "cirrhosis"),
                   "auc_pc_conf"].iloc[0]
    )

    return {
        "gene": gene,
        "mr_cr_drv": mr_cr_drv, "mr_cr_prt": mr_cr_prt,
        "SE_EXP": se_exp,
        "b_pos": b_pos, "b_neg": b_neg, "b_p": b_p, "b_log2fc": b_log2fc,
        "mr_serpine": mr_serpine,
        "se_serpine": se_serpine,
        "beta_exp_serpine": beta_exp_serpine,
        "val": val, "gtex_incohort_auc": gtex_auc,
    }

# Palette for the steatosis MR funnel (Fig S6A) — matches drivers orange
FUNNEL_STEAT = ["#FBE9D6", "#F4CCA1", "#E19137", "#B96A18"]

def draw_mr_funnel(fig, ax, mr_drv, mr_prt, palette, title,
                   n_top=100):
    """Four-stage MR triage funnel — used by main Fig 4A and supp S6A."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    olink = mr_drv["Gene"].nunique()
    outcome_n = mr_drv[mr_drv["p_out"] < 0.05]["Gene"].nunique()
    prot_n = mr_prt[mr_prt["p_out"] < 0.05]["Gene"].nunique()

    stages = [
        (f"Top {n_top} blood FC (upregulated)", n_top),
        ("Olink + cis-pQTL",                   olink),
        ("Outcome significant (p < 0.05)",     outcome_n),
        ("Driver candidates",                  outcome_n),
    ]
    widths = np.linspace(0.98, 0.55, len(stages))
    heights, top, gap = 0.14, 0.88, 0.03
    y_positions = []
    for i, ((lbl, val), w_row, col) in enumerate(zip(stages, widths, palette)):
        y_top = top - i * (heights + gap); y_bot = y_top - heights
        x0 = 0.5 - w_row / 2
        y_positions.append((y_top, y_bot))
        ax.add_patch(FancyBboxPatch(
            (x0, y_bot), w_row, heights,
            boxstyle="round,pad=0.005,rounding_size=0.015",
            linewidth=0.6, edgecolor=INK, facecolor=col, zorder=2))
        text_color = "white" if i >= 2 else INK
        ax.text(0.5, y_bot + heights * 0.62, lbl,
                ha="center", va="center", fontsize=8.5, color=text_color,
                zorder=3)
        ax.text(0.5, y_bot + heights * 0.25, f"n = {val}",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color=text_color, zorder=3)
    for i in range(len(stages) - 1):
        ax.add_patch(FancyArrowPatch(
            (0.5, y_positions[i][1]), (0.5, y_positions[i + 1][0]),
            arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color=INK,
            zorder=1))
    y_bot_final = y_positions[-1][1]
    ax.text(0.5, y_bot_final - 0.03,
            f"Protective candidates: n = {prot_n}",
            ha="center", va="top", fontsize=8.5, fontweight="bold",
            color=NEUTRAL_GREY)
    ax.text(0.5, 0.97, title,
            ha="center", va="top", fontsize=10, fontweight="bold",
            color=INK, transform=ax.transAxes)
    ax.set_ylim(y_bot_final - 0.10, 1.0)

def panel_fig4A(fig, ax, data):
    """MR triage funnel for Liver – Cirrhosis."""
    draw_mr_funnel(
        fig, ax,
        mr_drv=data["mr_cr_drv"], mr_prt=data["mr_cr_prt"],
        palette=FUNNEL_CIRR,
        title=f"Liver {EN_DASH} Cirrhosis MR triage",
    )

def panel_fig4B(fig, ax, data):
    """SERPINE1 whole-blood TPM boxplot — NB16 (Fig 6) design."""
    b_pos = data["b_pos"]; b_neg = data["b_neg"]
    b_p = data["b_p"]; b_log2fc = data["b_log2fc"]; gene = data["gene"]
    COL_POS, COL_NEG = CORAL, STEEL

    bp = ax.boxplot([b_neg, b_pos], positions=[0, 1], widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.5),
                    boxprops=dict(lw=0.8), whiskerprops=dict(lw=0.8),
                    capprops=dict(lw=0.8))
    for patch, col in zip(bp["boxes"], [COL_NEG, COL_POS]):
        patch.set_facecolor(col); patch.set_alpha(0.7); patch.set_edgecolor("black")

    rng = np.random.default_rng(0)
    for i, (vals, col) in enumerate(zip([b_neg, b_pos], [COL_NEG, COL_POS])):
        jx = rng.normal(loc=i, scale=0.06, size=len(vals))
        ax.scatter(jx, vals, s=8, color=col, alpha=0.6,
                   edgecolor="black", linewidth=0.3, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"No Cirrhosis\n(n={len(b_neg)})",
                        f"Cirrhosis\n(n={len(b_pos)})"], fontsize=10)
    ax.set_ylabel(f"Whole-blood {gene} TPM", fontsize=11)
    ax.set_yscale("log")
    ax.tick_params(axis="y", labelsize=10)

    _p_str = f"p = {b_p:.2e}" if b_p < 1e-3 else f"p = {b_p:.3f}"
    ax.text(0.5, 0.98, f"log$_2$FC = {b_log2fc:+.2f}\n{_p_str}",
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))
    thin_spines(ax)

def panel_fig4C(fig, ax, data):
    """SERPINE1 cis-pQTL exposure instrument — NB16 (Fig 6B) design."""
    beta_exp = data["beta_exp_serpine"]
    se_exp = data["se_serpine"]
    gene = data["gene"]
    COL_POS = CORAL

    ax.errorbar([0], [beta_exp],
                yerr=[[1.96 * se_exp], [1.96 * se_exp]],
                fmt="o", color=COL_POS, markersize=9,
                markeredgecolor="black", markeredgewidth=0.6,
                ecolor="black", elinewidth=1.0, capsize=6, capthick=1.0)
    ax.set_xticks([0])
    ax.set_xticklabels([gene], fontsize=11, fontweight="bold")
    ax.set_ylabel("cis-pQTL effect on plasma protein\n(UKB-PPP, SD units)",
                  fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", pad=1, length=0)
    ax.set_xlim(-0.7, 0.7)

    ci_lo = beta_exp - 1.96 * se_exp
    ci_hi = beta_exp + 1.96 * se_exp
    ax.set_ylim(-0.14, ci_hi + 0.015)
    ax.set_yticks([-0.14, -0.13, -0.12, -0.11, -0.10, -0.09])

    ax.text(0.5, 0.90,
            f"β = {beta_exp:+.3f}\n95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]",
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
            fontweight="bold", color=INK)
    thin_spines(ax)

def _sort_by_severity(mr, order=PHEN_ORDER_SEVERITY):
    ord_map = {p: i for i, p in enumerate(order)}
    d = mr.copy()
    d["_ord"] = d["Phenotype"].map(ord_map)
    d = d.sort_values(["_ord", "p_MR"], ascending=[True, True])
    return d.iloc[::-1].reset_index(drop=True)

def panel_fig4D(fig, ax, data):
    """SERPINE1 β_out forest across 6 FinnGen phenotypes — approved Fig 4D."""
    mr_serpine = data["mr_serpine"]
    PHEN_ORDER = ["NAFLD", "NASH", "CHIRHEP_NAS", "CIRRHOSIS_BROAD",
                  "FIBROLIV", "K11_FIBROCHIRLIV"]
    PHEN_LABEL_FULL = {
        "NAFLD":            "NAFLD",
        "NASH":             "NASH",
        "CIRRHOSIS_BROAD":  "Broad\nCirrhosis",
        "K11_FIBROCHIRLIV": "Fibrosis /\nCirrhosis",
        "CHIRHEP_NAS":      "Chronic\nHepatitis",
        "FIBROLIV":         "Liver\nFibrosis",
    }
    COL_POS = CORAL
    GRAY_MUT = "#888888"

    mr_ord = (mr_serpine.set_index("Phenotype")
              .reindex(PHEN_ORDER).reset_index())
    y = np.arange(len(mr_ord))
    for yi, (_, row) in enumerate(mr_ord.iterrows()):
        b, se, pv = row.beta_out, row.SE, row.p_MR
        col = COL_POS if pv < 0.05 else GRAY_MUT
        ax.plot([b - 1.96 * se, b + 1.96 * se], [yi, yi], color="0.4", lw=0.9)
        ax.plot(b, yi, "o", color=col, markersize=8,
                markeredgecolor="black", markeredgewidth=0.4, zorder=3)

    ax.axvline(0, color="black", ls="--", lw=0.6, alpha=0.4, zorder=0)

    xlim_hi = float(np.nanmax([row.beta_out + 1.96 * row.SE
                               for _, row in mr_ord.iterrows()])) + 0.30
    for yi, (_, row) in enumerate(mr_ord.iterrows()):
        ax.text(xlim_hi - 0.02, yi, f"p = {row.p_MR:.3f}",
                ha="right", va="center", fontsize=9, color="0.30")

    ax.set_yticks(y)
    ax.set_yticklabels([PHEN_LABEL_FULL[p] for p in mr_ord.Phenotype],
                       fontsize=10)
    ax.set_xlabel("cis-pQTL effect on disease\n(FinnGen R12, log-odds)",
                  fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(ax.get_xlim()[0], xlim_hi)
    thin_spines(ax)

def panel_fig4E(fig, ax, data):
    """External-validation AUC — approved Fig 4E (3-bar comparison)."""
    val = data["val"]
    gtex_auc = data["gtex_incohort_auc"]
    # K = 100 matches the approved Jul-15 values (0.80 healthy / 0.64 septic)
    top = val[val["K"] == 100].iloc[0]
    auc_healthy = float(top["AUC_vs_healthy_only"])
    auc_septic = float(top["AUC_vs_septicshock_only"])

    bar_labels = [
        "GTEx\nin-cohort",
        "External\nCirrhosis vs\nHealthy",
        "External\nCirrhosis vs\nSeptic Shock",
    ]
    aucs = [gtex_auc, auc_healthy, auc_septic]
    colors = ["#888888", CORAL, STEEL]

    x = np.arange(len(bar_labels))
    bars = ax.bar(x, aucs, color=colors, edgecolor="none", width=0.6)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=INK)
    ax.axhline(0.5, color=NEUTRAL_GREY, lw=0.7, ls="--", alpha=0.6, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.set_ylabel("AUC", fontsize=11, fontweight="bold")
    ax.set_ylim(0.4, 1.0)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlabel("External: GSE142255 (Weiss et al. 2021)", fontsize=9)
    thin_spines(ax)

_COL_DRV = "#E99A00"   # orange (Causal Drivers)
_COL_PRT = "#A64DB3"   # purple (Causal Protective)

_PHEN_LABELS_FG = {
    "K11_FIBROCHIRLIV": "K11\nFIBROCHIRLIV",
    "CIRRHOSIS_BROAD":  "CIRRHOSIS\n(BROAD)",
    "FIBROLIV":         "FIBROLIVER",
    "CHIRHEP_NAS":      "CHIRHEP\nNAS",
    "NASH":             "NASH",
    "NAFLD":            "NAFLD",
}

# Manual gene/phenotype ordering — visual order matches approved artwork.
# Lists are top-to-bottom; the loader reverses them for matplotlib's y=0-bottom.
_DRIVER_ORDER = [
    ("K11_FIBROCHIRLIV", "TIMP4"),
    ("K11_FIBROCHIRLIV", "CEACAM6"),
    ("K11_FIBROCHIRLIV", "TNC"),
    ("CIRRHOSIS_BROAD",  "TIMP4"),
    ("CIRRHOSIS_BROAD",  "DNAJC6"),
    ("CIRRHOSIS_BROAD",  "C7"),
    ("FIBROLIV",         "SERPINE1"),
    ("FIBROLIV",         "SLC4A1"),
    ("FIBROLIV",         "FKBP1B"),
    ("FIBROLIV",         "CD177"),
    ("CHIRHEP_NAS",      "DNAJC6"),
    ("CHIRHEP_NAS",      "CEACAM6"),
    ("NASH",             "THBS2"),
    ("NASH",             "TNC"),
    ("NASH",             "TFF1"),
    ("NAFLD",            "ASGR2"),
    ("NAFLD",            "ACHE"),
    ("NAFLD",            "CREG1"),
    ("NAFLD",            "CCL24"),
]

_PROTECTIVE_ORDER = [
    ("K11_FIBROCHIRLIV", "CFB"),
    ("K11_FIBROCHIRLIV", "SERPINC1"),
    ("CIRRHOSIS_BROAD",  "CFB"),
    ("FIBROLIV",         "FABP1"),
    ("FIBROLIV",         "FCER2"),
    ("FIBROLIV",         "GZMH"),
    ("CHIRHEP_NAS",      "CFB"),
    ("NASH",             "CXCL8"),
    ("NASH",             "SULT2A1"),
    ("NASH",             "SERPINC1"),
    ("NAFLD",            "GBP4"),
]

def _order_rows(frame, order):
    """Pick rows in the given (Phenotype, Gene) order and reverse for mpl."""
    rows = []
    for phenotype, gene in order:
        hit = frame[(frame["Phenotype"] == phenotype)
                    & (frame["Gene"] == gene)]
        if not hit.empty:
            rows.append(hit.iloc[0])
    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)

# Phenotype visual order used by the auto-orderer — top to bottom.
_MR_PHEN_VISUAL_ORDER = ["K11_FIBROCHIRLIV", "CIRRHOSIS_BROAD",
                        "FIBROLIV", "CHIRHEP_NAS", "NASH", "NAFLD"]

def order_rows_auto(frame, sig_col="p_MR", sig_thresh=0.05,
                    gene_sort_col="p_MR"):
    """Auto-order MR rows by phenotype block for `draw_causal_panel`."""
    sig = frame[frame[sig_col] < sig_thresh].copy()
    if sig.empty:
        return sig.reset_index(drop=True)
    ord_map = {p: i for i, p in enumerate(_MR_PHEN_VISUAL_ORDER)}
    sig["_ord"] = sig["Phenotype"].map(ord_map).fillna(len(ord_map))
    sig = sig.sort_values(["_ord", gene_sort_col],
                          ascending=[True, True]).reset_index(drop=True)
    sig = sig.drop(columns=["_ord"])
    return sig.iloc[::-1].reset_index(drop=True)

def draw_causal_panel(fig, gs_slot, frame, color, title, subtitle, fibro_label,
                      exp_xlim=None, exp_xticks=None,
                      out_xlim=None, out_xticks=None,
                      k11_strip_text=None,
                      strip_width_ratio=0.82):
    """[strip | β_exp | gap | β_out] — recovered exact design."""
    inner = gs_slot.subgridspec(
        1, 4, width_ratios=[strip_width_ratio, 1.0, 0.035, 1.0], wspace=0.0)
    ax_strip = fig.add_subplot(inner[0, 0])
    ax_exp   = fig.add_subplot(inner[0, 1])
    ax_gap   = fig.add_subplot(inner[0, 2])
    ax_gap.axis("off")
    ax_out   = fig.add_subplot(inner[0, 3], sharey=ax_exp)

    n = len(frame)
    y = np.arange(n)

    starts = [0]
    for i in range(1, n):
        if frame.iloc[i]["Phenotype"] != frame.iloc[i - 1]["Phenotype"]:
            starts.append(i)
    starts.append(n)

    for k in range(len(starts) - 1):
        s, e = starts[k], starts[k + 1] - 1
        phen = frame.iloc[s]["Phenotype"]
        ax_strip.axhspan(s - 0.5, e + 0.5, color="#E2E2E2", zorder=0)

        # Drivers panel omits the K11 prefix; both panels remap FIBROLIV.
        if phen == "K11_FIBROCHIRLIV":
            if k11_strip_text is not None:
                strip_text = k11_strip_text
            else:
                strip_text = ("FIBROCHIRLIV" if color == _COL_DRV
                              else _PHEN_LABELS_FG[phen])
        elif phen == "FIBROLIV":
            strip_text = fibro_label
        else:
            strip_text = _PHEN_LABELS_FG[phen]

        ax_strip.text(0.055, (s + e) / 2, strip_text,
                      ha="left", va="center", fontsize=10.0,
                      fontweight="bold", color="#111111", linespacing=0.98)

        if e + 1 < n:
            ax_strip.axhline(e + 0.5, color="white", lw=1.0)

    for yi, (_, row) in enumerate(frame.iterrows()):
        ax_exp.plot(row["beta_exp"], yi, "o", color=color, ms=9.0,
                    mec="black", mew=0.35)
        ax_out.errorbar(row["beta_out"], yi, xerr=1.96 * row["SE"],
                        fmt="none", ecolor="#444444", elinewidth=1.15,
                        capsize=2.2, capthick=1.1, zorder=4)
        ax_out.plot(row["beta_out"], yi, "o", color=color, ms=9.0,
                    mec="black", mew=0.35)

    for ax in (ax_exp, ax_out):
        ax.axvline(0, color="#555555", ls="--", lw=0.8, alpha=0.75)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_axisbelow(True)
        ax.grid(True, color="#E5E5E5", lw=0.65, linestyle="-")
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#333333")
            spine.set_linewidth(0.7)

    ax_exp.set_yticks(y)
    ax_exp.set_yticklabels(frame["Gene"], fontsize=11)
    ax_exp.tick_params(axis="y", pad=5, length=4, width=0.8,
                       direction="out", color="#555555")
    ax_out.tick_params(axis="y", labelleft=False)

    ax_exp.set_xlabel(
        "cis-pQTL effect on plasma protein\n(UKB-PPP, SD units)",
        fontsize=10.0)
    ax_out.set_xlabel(
        "cis-pQTL effect on disease\n(FinnGen R12, log-odds)",
        fontsize=10.0)

    def _autolim(values, pad_frac=0.15):
        vmin = float(np.nanmin(values)); vmax = float(np.nanmax(values))
        span = vmax - vmin if vmax > vmin else max(abs(vmin), abs(vmax), 0.1)
        pad = span * pad_frac
        lo, hi = min(vmin - pad, 0.0), max(vmax + pad, 0.0)
        return lo, hi

    if exp_xlim is None:
        ax_exp.set_xlim(*_autolim(frame["beta_exp"].values))
    else:
        ax_exp.set_xlim(*exp_xlim)
    if exp_xticks is not None:
        ax_exp.set_xticks(exp_xticks)

    out_hi = (frame["beta_out"] + 1.96 * frame["SE"]).values
    out_lo = (frame["beta_out"] - 1.96 * frame["SE"]).values
    if out_xlim is None:
        ax_out.set_xlim(*_autolim(np.concatenate([out_lo, out_hi])))
    else:
        ax_out.set_xlim(*out_xlim)
    if out_xticks is not None:
        ax_out.set_xticks(out_xticks)

    ax_strip.set_xlim(0, 1)
    ax_strip.set_ylim(-0.5, n - 0.5)
    ax_strip.axis("off")

    pos_exp = ax_exp.get_position()
    pos_out = ax_out.get_position()
    center = (pos_exp.x0 + pos_out.x1) / 2
    fig.text(center, pos_exp.y1 + 0.036, title,
             ha="center", va="bottom", fontsize=15, fontweight="bold")
    fig.text(center, pos_exp.y1 + 0.010, f"({subtitle})",
             ha="center", va="bottom", fontsize=11.5, fontweight="bold")
    return ax_strip, ax_exp, ax_out

def panel_fig4F(fig, gs_slot, data):
    """Causal Drivers strip — manual gene/phenotype ordering, recovered design."""
    drv = _order_rows(data["mr_cr_drv"], _DRIVER_ORDER)
    return draw_causal_panel(
        fig, gs_slot, drv, _COL_DRV,
        title="Causal Drivers",
        subtitle="upregulated genes + same-direction MR",
        fibro_label="FIBROCHIRLIV",
        exp_xlim=(-1.7, 0.9), exp_xticks=[-1.0, 0.0],
        out_xlim=(-1.85, 0.8), out_xticks=[-1.0, 0.0],
    )

def panel_fig4G(fig, gs_slot, data):
    """Causal Protective strip — manual gene/phenotype ordering, recovered."""
    prt = _order_rows(data["mr_cr_prt"], _PROTECTIVE_ORDER)
    return draw_causal_panel(
        fig, gs_slot, prt, _COL_PRT,
        title="Causal Protective",
        subtitle="downregulated genes + opposite-direction MR",
        fibro_label="FIBROLIVER",
        exp_xlim=(-1.1, 0.8), exp_xticks=[-1.0, -0.5, 0.0, 0.5],
        out_xlim=(-0.4, 3.8), out_xticks=[0.0, 1.0, 2.0, 3.0],
    )

