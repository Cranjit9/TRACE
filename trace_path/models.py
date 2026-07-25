"""Cross-validation pipelines — shared by LR and RF models."""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, roc_auc_score

from trace_path.config import Config

def _direction_agnostic_auc(scores, binary_target):
    """max(AUC, 1-AUC); returns 0.5 if target degenerate or n<5."""
    s = pd.to_numeric(scores, errors="coerce")
    t = pd.to_numeric(binary_target, errors="coerce")
    mask = s.notna() & t.notna()
    if mask.sum() < 5:
        return 0.5
    tt = t[mask].astype(int)
    if tt.nunique() < 2:
        return 0.5
    auc = roc_auc_score(tt, s[mask])
    return float(max(auc, 1.0 - auc))

def _binarize_at_median(s):
    """Binarize a continuous Series at its median (within the input subset)."""
    s = pd.to_numeric(s, errors="coerce")
    med = s.median()
    return (s > med).astype(int)

def _compute_pc_covariate_aucs(pcs_df, X_covar):
    """Per-PC direction-agnostic AUC against each of the 5 clinical covariates."""
    # Recipe locked 2026-06-22: SEX direct; AGE/TRISCHD binarize-at-median;
    # RACE and DTHHRDY one-vs-rest max AUC across classes.
    out = pd.DataFrame(
        index=pcs_df.columns,
        columns=["AGE", "SEX", "RACE", "DTHHRDY", "TRISCHD"],
        dtype=float,
    )

    sex = pd.to_numeric(X_covar["SEX"], errors="coerce")
    age_b = _binarize_at_median(X_covar["AGE"])
    trischd_b = _binarize_at_median(X_covar["TRISCHD"])

    for pc in pcs_df.columns:
        pc_vals = pcs_df[pc]
        out.at[pc, "SEX"] = _direction_agnostic_auc(pc_vals, sex)
        out.at[pc, "AGE"] = _direction_agnostic_auc(pc_vals, age_b)
        out.at[pc, "TRISCHD"] = _direction_agnostic_auc(pc_vals, trischd_b)

    for col_name in ("RACE", "DTHHRDY"):
        col = X_covar[col_name]
        levels = sorted({float(v) for v in col.dropna().unique()})
        best = pd.Series(0.5, index=pcs_df.columns, dtype=float)
        for lvl in levels:
            tgt = (col == lvl).astype(int)
            if tgt.sum() < 5 or (1 - tgt).sum() < 5:
                continue
            for pc in pcs_df.columns:
                a = _direction_agnostic_auc(pcs_df[pc], tgt)
                if a > best[pc]:
                    best[pc] = a
        out[col_name] = best

    return out

def _drop_covariate_aligned_pcs(pc_aucs_per_covar, threshold):
    """Boolean keep mask: True iff max AUC across covariates <= threshold."""
    return pc_aucs_per_covar.max(axis=1) <= threshold

def _auc_feature_selection(Xtr, ytr, top_k=None):
    """Per-gene AUC-based feature selection on training data only."""
    top_k = top_k or Config.TOP_K_FEATURES
    strengths = {}
    for col in Xtr.columns:
        s = pd.to_numeric(Xtr[col], errors="coerce")
        med = s.median()
        s = s.fillna(med if pd.notna(med) else 0.0)
        if s.nunique() < 2:
            strengths[col] = 0.0
            continue
        a = roc_auc_score(ytr, s)
        strengths[col] = abs(a - 0.5)
    scores = pd.Series(strengths).reindex(Xtr.columns).fillna(0)
    return scores.nlargest(top_k).index.tolist()

def _impute_train_fold(Xtr, Xte):
    """Median-impute Xtr (using train-fold medians) and apply same medians to Xte."""
    med = Xtr.median(numeric_only=True)
    med = med.where(med.notna(), 0.0)  # all-NaN train columns fall back to 0
    return Xtr.fillna(med), Xte.fillna(med)

def make_lr_pipeline(cfg=None):
    """Create a Logistic Regression pipeline with StandardScaler."""
    cfg = cfg or Config
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            solver=cfg.LR_SOLVER, max_iter=cfg.LR_MAX_ITER,
            random_state=cfg.SEED,
        )),
    ])

def make_rf_model(cfg=None):
    """Create a Random Forest classifier (no scaler needed)."""
    cfg = cfg or Config
    return RandomForestClassifier(
        n_estimators=cfg.RF_N_ESTIMATORS,
        max_features=cfg.RF_MAX_FEATURES,
        class_weight="balanced",
        random_state=cfg.SEED,
        n_jobs=1,  # n_jobs=1 inside parallel workers to avoid oversubscription
    )

def _compute_oof_threshold(y, oof):
    """Compute Youden's J optimal threshold from OOF predictions."""
    mask = ~np.isnan(oof)
    fpr_oof, tpr_oof, thresholds_oof = roc_curve(
        y.values[mask], oof[mask]
    )
    j_scores = tpr_oof - fpr_oof
    return thresholds_oof[np.argmax(j_scores)]

def run_cv(X, y, groups, model_factory, cfg=None, top_k=None,
           save_features=False, X_extra=None, feature_selection=True):
    """Run leak-free 5-fold grouped CV."""
    cfg = cfg or Config
    top_k = top_k or cfg.TOP_K_FEATURES

    cv = StratifiedGroupKFold(
        n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED
    )

    oof = np.full(len(y), np.nan)
    fold_fprs, fold_tprs, fold_aucs = [], [], []
    feature_info = [] if save_features else None

    for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        if feature_selection:
            top_feat = _auc_feature_selection(Xtr, ytr, top_k=top_k)
            Xtr_fit = Xtr[top_feat]
            Xte_fit = Xte[top_feat]
        else:
            top_feat = list(X.columns)
            Xtr_fit = Xtr
            Xte_fit = Xte

        if X_extra is not None:
            Xtr_extra, Xte_extra = X_extra.iloc[tr], X_extra.iloc[te]
            Xtr_fit = pd.concat([Xtr_fit, Xtr_extra], axis=1)
            Xte_fit = pd.concat([Xte_fit, Xte_extra], axis=1)

        # Per-fold median imputation avoids test-fold leakage.
        Xtr_fit, Xte_fit = _impute_train_fold(Xtr_fit, Xte_fit)

        model = model_factory()
        model.fit(Xtr_fit, ytr)
        proba = model.predict_proba(Xte_fit)[:, 1]
        oof[te] = proba

        fauc = roc_auc_score(yte, proba)
        fpr, tpr, _ = roc_curve(yte, proba)
        fold_fprs.append(fpr)
        fold_tprs.append(tpr)
        fold_aucs.append(fauc)

        if save_features:
            estimator = (model.named_steps["model"]
                         if hasattr(model, "named_steps") else model)
            importances = getattr(estimator, "feature_importances_", None)
            fold_info = {
                "fold": fold,
                "selected_genes": top_feat,
                "rf_importances": (dict(zip(top_feat, importances))
                                   if importances is not None else {}),
            }
            feature_info.append(fold_info)

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    optimal_thresh = _compute_oof_threshold(y, oof)

    result = {
        "y": y, "oof": oof,
        "fold_fprs": fold_fprs, "fold_tprs": fold_tprs, "fold_aucs": fold_aucs,
        "mean_auc": mean_auc, "std_auc": std_auc,
        "optimal_threshold": optimal_thresh,
    }
    if save_features:
        result["feature_info"] = feature_info
    return result

def run_cv_no_fs(X, y, groups, model_factory, cfg=None):
    """Run CV without feature selection (for low-dimensional inputs like clinical co-variates)."""
    return run_cv(X, y, groups, model_factory, cfg=cfg, feature_selection=False)

def run_cv_combined(X_expr, X_conf, y, groups, model_factory, cfg=None, top_k=None):
    """Run CV with AUC FS on expression, always including clinical co-variates."""
    return run_cv(X_expr, y, groups, model_factory, cfg=cfg, top_k=top_k,
                  X_extra=X_conf)

def run_tissue_models(tissue, cat_list, df_meta_url, blood_subjid, X_wb,
                      model_factory, cfg=None, save_features=False):
    """Run CV for all categories of a single tissue (one call per tissue inside joblib.Parallel)."""
    from trace_path.labels import assign_donor_labels

    cfg = cfg or Config
    results = []

    for cat, n_samples in cat_list:
        tag = f"{tissue} | {cat}"

        y, donor_lab, n_pos, n_neg = assign_donor_labels(
            df_meta_url, tissue, cat, blood_subjid
        )
        keep = y.notna()
        X_cat = X_wb.loc[keep].copy()
        y_cat = y.loc[keep].astype(int)
        g_cat = blood_subjid.loc[keep].astype(str)

        if n_pos < cfg.MIN_POS_NEG_BLOOD or n_neg < cfg.MIN_POS_NEG_BLOOD:
            continue

        res = run_cv(X_cat, y_cat, g_cat, model_factory, cfg=cfg,
                     save_features=save_features)
        res["tissue"] = tissue
        res["category"] = cat
        results.append((tag, res))

    return results

def run_cv_with_pca(X, y, groups, n_pcs=800, top_k_pcs=100, X_conf_sub=None,
                    cfg=None, tag="", model_factory=None,
                    X_covar=None, ortho_thresholds=None):
    """Leak-free CV with per-fold PCA, optionally sweeping covariate-orthogonalization thresholds."""
    cfg = cfg or Config
    if model_factory is None:
        model_factory = make_rf_model

    sweep_mode = ortho_thresholds is not None and len(ortho_thresholds) > 0
    if sweep_mode and X_covar is None:
        raise ValueError("X_covar is required when ortho_thresholds is set.")

    cv = StratifiedGroupKFold(
        n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED
    )

    if sweep_mode:
        per_thresh = {t: {"oof": np.full(len(y), np.nan), "fold_aucs": []}
                      for t in ortho_thresholds}
        dropped_records = []
    else:
        oof_legacy = np.full(len(y), np.nan)
        fold_aucs_legacy = []

    for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        scaler = StandardScaler()
        Xtr_scaled = scaler.fit_transform(Xtr)
        Xte_scaled = scaler.transform(Xte)

        n_comp = min(n_pcs, Xtr_scaled.shape[0] - 1, Xtr_scaled.shape[1])
        pca = PCA(n_components=n_comp, random_state=cfg.SEED)
        Xtr_pcs_arr = pca.fit_transform(Xtr_scaled)
        Xte_pcs_arr = pca.transform(Xte_scaled)

        pc_cols = [f"PC{i+1}" for i in range(n_comp)]
        Xtr_pcs = pd.DataFrame(Xtr_pcs_arr, columns=pc_cols, index=Xtr.index)
        Xte_pcs = pd.DataFrame(Xte_pcs_arr, columns=pc_cols, index=Xte.index)

        # Threshold-independent — compute once per fold.
        if sweep_mode:
            X_covar_tr = X_covar.iloc[tr]
            pc_aucs = _compute_pc_covariate_aucs(Xtr_pcs, X_covar_tr)

        thresholds_iter = ortho_thresholds if sweep_mode else [None]
        for thresh in thresholds_iter:
            if sweep_mode:
                keep_mask = _drop_covariate_aligned_pcs(pc_aucs, thresh)
                surviving = pc_aucs.index[keep_mask].tolist()
                if len(surviving) < top_k_pcs:
                    # Edge case: too many PCs dropped — fall through and use all survivors.
                    pass
                Xtr_pool = Xtr_pcs[surviving]
                Xte_pool = Xte_pcs[surviving]
                dropped_records.append({
                    "fold": fold,
                    "threshold": thresh,
                    "n_dropped": int((~keep_mask).sum()),
                    "n_kept": int(keep_mask.sum()),
                    "tag": tag,
                })
            else:
                Xtr_pool = Xtr_pcs
                Xte_pool = Xte_pcs

            top_pcs = _auc_feature_selection(
                Xtr_pool, ytr,
                top_k=min(top_k_pcs, Xtr_pool.shape[1])
            )
            Xtr_sel = Xtr_pool[top_pcs]
            Xte_sel = Xte_pool[top_pcs]

            if X_conf_sub is not None:
                Xtr_sel = pd.concat([Xtr_sel, X_conf_sub.iloc[tr]], axis=1)
                Xte_sel = pd.concat([Xte_sel, X_conf_sub.iloc[te]], axis=1)
                Xtr_sel, Xte_sel = _impute_train_fold(Xtr_sel, Xte_sel)

            model = model_factory()
            model.fit(Xtr_sel, ytr)
            proba = model.predict_proba(Xte_sel)[:, 1]

            fauc = float(roc_auc_score(yte, proba))

            if sweep_mode:
                per_thresh[thresh]["oof"][te] = proba
                per_thresh[thresh]["fold_aucs"].append(fauc)
            else:
                oof_legacy[te] = proba
                fold_aucs_legacy.append(fauc)

    if sweep_mode:
        per_thresh_out = {}
        for t, d in per_thresh.items():
            arr = np.asarray(d["fold_aucs"])
            per_thresh_out[t] = {
                "mean_auc": float(arr.mean()),
                "std_auc": float(arr.std()),
                "fold_aucs": arr.tolist(),
                "oof": d["oof"],
            }
        if tag:
            line = f"    {tag} "
            line += " ".join(
                f"[t={t:.2f}: {per_thresh_out[t]['mean_auc']:.3f}±{per_thresh_out[t]['std_auc']:.3f}]"
                for t in ortho_thresholds
            )
            print(line)
        return {
            "thresholds": list(ortho_thresholds),
            "per_threshold": per_thresh_out,
            "dropped_pcs": dropped_records,
            "y": y,
        }
    else:
        arr = np.asarray(fold_aucs_legacy)
        if tag:
            print(f"    {tag} Mean AUC = {arr.mean():.3f} +/- {arr.std():.3f}")
        return {
            "mean_auc": float(arr.mean()),
            "std_auc": float(arr.std()),
            "fold_aucs": arr.tolist(),
            "oof": oof_legacy,
            "y": y,
        }

def run_tissue_confounder_models(tissue, cat_list, df_meta_url, blood_subjid,
                                 X_wb, X_conf, model_factory, cfg=None):
    """Run clinical co-variate-only AND expression+clinical co-variate models for one tissue."""
    from trace_path.labels import assign_donor_labels

    cfg = cfg or Config
    conf_results = []
    comb_results = []

    for cat, n_samples in cat_list:
        tag = f"{tissue} | {cat}"

        y, donor_lab, n_pos, n_neg = assign_donor_labels(
            df_meta_url, tissue, cat, blood_subjid
        )
        keep = y.notna()
        X_expr_cat = X_wb.loc[keep].copy()
        X_conf_cat = X_conf.loc[keep].copy()
        y_cat = y.loc[keep].astype(int)
        g_cat = blood_subjid.loc[keep].astype(str)

        if n_pos < cfg.MIN_POS_NEG_BLOOD or n_neg < cfg.MIN_POS_NEG_BLOOD:
            continue

        res_c = run_cv(X_conf_cat, y_cat, g_cat, model_factory, cfg=cfg,
                       feature_selection=False)
        res_c["tissue"] = tissue
        res_c["category"] = cat
        conf_results.append((tag, res_c))

        res_cb = run_cv(X_expr_cat, y_cat, g_cat, model_factory, cfg=cfg,
                        X_extra=X_conf_cat)
        res_cb["tissue"] = tissue
        res_cb["category"] = cat
        comb_results.append((tag, res_cb))

    return conf_results, comb_results
