"""Donor-level labelling and NLP-based pathology imputation."""

import re
import pandas as pd
import numpy as np
from collections import Counter

from trace_path.config import Config

CATEGORY_PATTERNS = {
    "steatosis":      r"steatosis|steatotic|macrovesicular\s+steat|microvesicular\s+steat",
    "congestion":     r"congest|congnest|sinusoidal dilat",
    "fibrosis":       r"fibros|fibrous|fibrotic|bridging",
    "cirrhosis":      r"cirrh",
    "inflammation":   r"inflam|lymphocyte|lymphoid\s+infiltrat|infiltrat",
    "necrosis":       r"necrosi|necrotic",
    "hepatitis":      r"hepatitis",
    "atrophy":        r"atroph",
    "hemorrhage":     r"hemorrhag|haemorrhag",
    "nodularity":     r"nodul",
    "hyperplasia":    r"hyperplas",
    "sclerotic":      r"scleros|sclerotic",
    "pigment":        r"pigment|lipofuscin|hemosiderin",
    "no_abnormalities": r"no abnormal|no\s+major\s+abnormal|within normal|unremarkable",
    "clean_specimens":  r"\bclean\b|no lesion|good specimens?|excellent specimens?",
}

_CLAUSE_SPLIT = re.compile(r"[.;]")
_SCOPE_TERM_SPLIT = re.compile(
    r"\b(?:and|but|however|yet|although|except|presenting|presents)\b"
    r"|consistent\s+with|\bc/w\b",
    re.IGNORECASE,
)
_POSITIVE_QUALIFIER = re.compile(
    r"^\s*(mild|moderate|severe|marked|significant|diffuse|focal|minimal"
    r"|prominent|central|passive|active|chronic|acute|slight|extensive"
    r"|occasional|few|several|some|scattered|moderately|focally|diffusely"
    r"|markedly|mildly|slightly|predominantly|unremarkable)",
    re.IGNORECASE,
)
_NEGATION_TRIGGER = re.compile(
    r"\b(no|not|without|absent|absence\s+of|negative|denies|deny"
    r"|free\s+of|ruled\s+out|rules\s+out|unlikely)\b",
    re.IGNORECASE,
)

# Pre-compiled patterns (compiled once at import time, not per row)
COMPILED_PATTERNS = {
    cat: re.compile(pat, re.IGNORECASE)
    for cat, pat in CATEGORY_PATTERNS.items()
}

def _is_negated_in_subclause(text, match_start):
    """Check if a match position is negated within its subclause."""
    subclauses = _SCOPE_TERM_SPLIT.split(text[:match_start + 20])
    last_sub = subclauses[-1] if subclauses else text
    if _POSITIVE_QUALIFIER.search(last_sub):
        return False
    return bool(_NEGATION_TRIGGER.search(last_sub))

def _smart_comma_split(text):
    """Split text on commas but respect parenthetical expressions."""
    parts = []
    depth = 0
    current = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    parts.append("".join(current).strip())
    return [p for p in parts if p]

def extract_categories(notes_text, compiled_patterns=None):
    """Extract pathology categories from free-text notes using regex + negation."""
    if compiled_patterns is None:
        compiled_patterns = COMPILED_PATTERNS

    if not isinstance(notes_text, str) or not notes_text.strip():
        return []

    matched = set()
    for clause in _CLAUSE_SPLIT.split(notes_text):
        for sub in _smart_comma_split(clause):
            sub = sub.strip()
            if not sub:
                continue
            for cat, rx in compiled_patterns.items():
                m = rx.search(sub)
                if m and not _is_negated_in_subclause(sub, m.start()):
                    matched.add(cat)

    # If real pathology found, drop normal labels
    if matched - Config.NORMAL_LABELS:
        matched -= Config.NORMAL_LABELS

    return sorted(matched)

# Column with structured-or-NLP-imputed labels. Added to df_meta_url by
# trace_path.data.load_raw_data(). Falls back to the raw structured column
# if the imputed column is absent (older cached data).
_LABEL_COL = "Pathology.Categories.Final"
_LABEL_COL_FALLBACK = "Pathology.Categories"

def _pick_label_col(df):
    if _LABEL_COL in df.columns:
        return _LABEL_COL
    return _LABEL_COL_FALLBACK

def assign_donor_labels(df_meta_url, tissue, category, blood_subjid):
    """Assign binary donor-level labels for a tissue × category pair."""
    label_col = _pick_label_col(df_meta_url)
    tissue_sub = df_meta_url[df_meta_url["Tissue"] == tissue].copy()
    tissue_sub["SUBJID"] = (
        tissue_sub["Tissue.Sample.ID"].astype(str)
        .str.split("-").str[:2].str.join("-")
    )

    known = tissue_sub[tissue_sub[label_col].notna()].copy()
    has_cat = known[label_col].str.contains(
        category, case=False
    ).astype(int)
    donor_lab = has_cat.groupby(known["SUBJID"]).max()

    y = blood_subjid.map(donor_lab)
    keep = y.notna()
    y_clean = y.loc[keep].astype(int)
    n_pos = int(y_clean.sum())
    n_neg = int((y_clean == 0).sum())

    return y, donor_lab, n_pos, n_neg

def discover_tissue_category_pairs(df_meta_url, threshold=None):
    """Find all tissue × category pairs with ≥ threshold positive samples."""
    threshold = threshold or Config.ALL_TISSUE_THRESHOLD
    exclude = {x.lower() for x in Config.NORMAL_LABELS}
    label_col = _pick_label_col(df_meta_url)
    pairs = []

    for tissue in sorted(df_meta_url["Tissue"].dropna().unique()):
        sub = df_meta_url[df_meta_url["Tissue"] == tissue]
        cat_counts = Counter()
        for val in sub[label_col].dropna():
            for c in val.split(", "):
                cat_counts[c.strip()] += 1
        for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
            if n >= threshold and cat.lower() not in exclude:
                pairs.append({"tissue": tissue, "category": cat, "n_samples": n})

    return pd.DataFrame(pairs)
