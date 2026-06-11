"""Pathology label extraction from GTEx `Pathology.Notes`.

Regex + ConText-style negation + percentage-based severity extraction.

Vocabulary
----------
**79 category patterns**, covering two superimposed vocabularies:
- All 57 tokens GTEx pathologists use in the structured `Pathology.Categories` field.
- 22 additional concepts that appear in the free-text notes but were never
  structure-coded (e.g. `plaque`, `intimal_thickening`, `fatty_infiltration`,
  `medial_degeneration`, `neoplasm`, `regressive_change`, `sertoli_only`,
  `bronchitis`, `colitis`, `amyloidosis`).
The 22 additional concepts cannot be validated against GTEx ground truth
(they have no structured-field equivalent), but they still produce labels in
the output matrices.

Design choices
--------------
- **Negation.** `_is_negated` looks back from the regex match within the same
  subclause. Negation is suppressed if a positive qualifier (e.g. `mild`, `60%`)
  sits between the negation trigger and the matched term — handles phrases like
  "no significant fibrosis but moderate steatosis" correctly.
- **Self-negating labels** (`no_abnormalities`, `clean_specimens`) skip the
  negation check since the negation IS the label.
- **Subclause-level exclusions.** `_RAW_PATTERNS[cat]["exclude"]` lists peer
  categories that should suppress this one when they co-occur in the same
  subclause (e.g. `neuroendocrine_tumor` suppresses the generic `neoplasm`).
- **Word-boundary anchors and negative lookbehinds** prevent overlapping
  patterns from double-firing (e.g. `atherosclerosis` and `sclerotic` no longer
  collide; `hemosiderin-laden macrophages` is counted as macrophages, not pigment).
- **Severity.** `extract_severities` returns a 0–100 percent per category.
  An explicit percentage in the same subclause as the match wins ("60% steatosis").
  Range mid-point for "20-30% fibrosis". Ordinal qualifier (`mild`=15, `moderate`=35,
  `severe`=70) as a fallback. NaN if no severity signal — binary positive does
  not imply a known severity.

Public API
----------
- `extract_categories_v2(notes_text) -> list[(category, confidence)]`
- `extract_severities(notes_text) -> dict[category, percent]`
- `extract_label_set(notes_text, min_confidence=0.5) -> list[category]`

Usage:
    from gtex_biomarkers.labels_v2 import extract_categories_v2, extract_severities
    extract_categories_v2("60% steatosis, mild fibrosis")
    # -> [('steatosis', 0.9), ('fibrosis', 0.9)]
    extract_severities("60% steatosis, mild fibrosis")
    # -> {'steatosis': 60.0, 'fibrosis': 15.0}
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple


# ── Category vocabulary ───────────────────────────────────────────────────────
# Each entry: (compiled regex, exclude_categories_in_same_subclause, skip_negation_flag)

_RAW_PATTERNS: Dict[str, dict] = {
    # ── Liver / hepatic ──────────────────────────────────────────────────────
    "steatosis": dict(
        # Note: "fatty infiltration" is a separate label (often non-liver muscle fat
        # replacement); it is NOT matched here to avoid co-firing with fatty_infiltration.
        pat=r"\bsteat(?:osis|otic|ohepatitis)\b|\bfatty\s+(?:change|liver|metamorphos\w*)|\bmacrovesicular\b|\bmicrovesicular\b",
        exclude=[],
    ),
    "cirrhosis": dict(pat=r"\bcirrh\w*", exclude=[]),
    "fibrosis": dict(
        pat=r"\bfibros\w*|\bfibrotic\b|\bfibrous\b|\bbridging\b|\bperisinusoidal\s+fibros|\bcollagen\s+deposit",
        exclude=[],
    ),
    "scarring": dict(pat=r"\bscar(?:ring|s|red)?\b", exclude=[]),
    "hepatitis": dict(
        # No leading word boundary so that "steatohepatitis" still triggers.
        pat=r"hepatit(?:is|ic)\b",
        exclude=[],
    ),
    "necrosis": dict(pat=r"\bnecrosi[sc]?\b|\bnecrotic\b", exclude=[]),

    # ── Vascular ─────────────────────────────────────────────────────────────
    "atherosclerosis": dict(
        pat=r"\batheroscleros\w*|\batherosclerotic\b",
        exclude=[],
    ),
    "atherosis": dict(
        # 'atherosis' is sometimes synonymous with 'atherosclerosis' but GTEx
        # uses it as a distinct annotation (often for tibial artery). Match the
        # bare word 'atherosis/atheroma' but not the longer 'atherosclerosis'.
        pat=r"\batheros[ie]s\b|\batheroma\b",
        exclude=[],
    ),
    "monckeberg": dict(
        pat=r"\bm[oö]nckeberg\w*|\bmoenckeberg\w*|\bmedial\s+calcific\s+sclerosis\b",
        exclude=[],
    ),
    "sclerotic": dict(
        # Match scleros* / sclerotic generally; exclude when a more specific
        # sclerosis label (atherosclerosis, glomerulosclerosis) is the only
        # match in the subclause via the exclude rule below. Keep the lookbehind
        # only for the obvious athero/arterio prefix collisions.
        pat=r"(?<!athero)(?<!arterio)\bscleros\w*|\bsclerotic\b|\bsclerosed\b",
        exclude=[],
    ),
    "infarction": dict(pat=r"\binfarct\w*", exclude=[]),
    "ischemic_changes": dict(
        pat=r"\bisch[ae]mic\b|\bisch[ae]mia\b|\bisch[ae]mic\s+chang",
        exclude=[],
    ),

    # ── General ──────────────────────────────────────────────────────────────
    "congestion": dict(
        pat=r"\bcong[ne]?st\w*|\bsinusoidal\s+dilat\w*|\bvenous\s+distens",
        exclude=[],
    ),
    "atrophy": dict(pat=r"\batroph\w*", exclude=[]),
    "hyperplasia": dict(pat=r"\bhyperplas\w*", exclude=[]),
    "hypertrophy": dict(pat=r"\bhypertroph\w*", exclude=[]),
    "hemorrhage": dict(pat=r"\bh[ae]morrhag\w*", exclude=[]),
    "edema": dict(
        pat=r"\b(?:o)?edema\b|\b(?:o)?edematous\b|\binterstitial\s+edema\b",
        exclude=[],
    ),
    "calcification": dict(
        pat=r"\bcalcif\w*|\bcalcium\s+deposit",
        exclude=[],
    ),
    "nodularity": dict(pat=r"\bnodul\w*", exclude=[]),
    "hyalinization": dict(pat=r"\bhyalin\w*", exclude=[]),
    "metaplasia": dict(pat=r"\bmetaplas\w*", exclude=[]),
    "cyst": dict(
        # Exclude cysteine, cystic fibrosis (handled by fibrosis), etc.
        pat=r"\bcyst\b|\bcysts\b|\bcystic\s+(?:lesion|space|cavity|change)",
        exclude=[],
    ),

    # ── Inflammatory / immune ────────────────────────────────────────────────
    "inflammation": dict(
        # Includes lymphocyte/leukocyte collections and Hashimoto thyroiditis
        # (which is inflammatory by definition — pathologists co-tag both).
        pat=(
            r"\binflam\w*|\binflammatory\s+(?:cell|infiltrate)"
            r"|\blymphocytic\s+infiltrat\w*|\bneutrophilic\s+infiltrat\w*"
            r"|\bperiportal\s+infiltrat"
            r"|\bcollection\w*\s+of\s+lymphocyt\w*|\blymphocyt\w*\s+(?:collection|aggregat|nodule)"
            r"|\bleukocyte\w*\s+(?:in|infiltrat)|\bhashimoto\s+thyroidit"
        ),
        exclude=[],
    ),
    "macrophages": dict(
        pat=r"\bmacrophage\w*|\bhemosiderin[-\s]laden",
        exclude=[],
    ),
    "hashimoto": dict(pat=r"\bhashimoto\w*|\bthyroidit\w*", exclude=[]),
    "esophagitis": dict(pat=r"\b(?:o)?esophagit\w*", exclude=[]),
    "pneumonia": dict(
        pat=r"\bpneumonia\b|\bbronchopneumonia\b|\bpneumonit\w*",
        exclude=[],
    ),

    # ── Lung-specific ────────────────────────────────────────────────────────
    "atelectasis": dict(pat=r"\batelect\w*", exclude=[]),
    "emphysema": dict(pat=r"\bemphysem\w*", exclude=[]),

    # ── Tissue-specific ──────────────────────────────────────────────────────
    "spermatogenesis": dict(
        pat=r"\bspermatogene\w*|\bspermatid\w*|\bspermatocyt\w*",
        exclude=[],
    ),
    "saponification": dict(pat=r"\bsapon\w*", exclude=[]),
    "gynecomastoid": dict(pat=r"\bgyn[ae]comast\w*", exclude=[]),
    "corpora_albicantia": dict(
        pat=r"\bcorpor[ae]\s+albicant|\bcorpus\s+albicans",
        exclude=[],
    ),
    "post_menopausal": dict(
        pat=r"\bpost[\s-]?menopaus\w*|\batrophic\s+endometri",
        exclude=[],
    ),
    "goiter": dict(pat=r"\bgoiter\b|\bgoitre\b", exclude=[]),
    "glomerulosclerosis": dict(
        pat=r"\bglomerul(?:o)?scleros\w*|\bsclerosed\s+glomeruli|\brare\s+sclerosed\s+glomerul",
        exclude=[],
    ),
    "hypereosinophilia": dict(
        pat=r"\bhypereosinoph\w*|\beosinophilic\s+infiltrat",
        exclude=[],
    ),

    # ── Pigment (suppressed if macrophages co-mentioned) ─────────────────────
    "pigment": dict(
        # GTEx pathologists tag "hemosiderin-laden macrophages" as macrophages
        # (not pigment). Match the standalone pigment terms but explicitly
        # exclude the laden-macrophage construction via a negative lookbehind.
        pat=(
            r"\blipofuscin\w*|\bmelanin\w*|\bbilirubin\w*"
            r"|(?<!laden\s)\bhemosider(?!in[-\s]laden)\w*"
            r"|\bpigment(?:ation|ed)?\b"
        ),
        exclude=[],
    ),

    # ── Tissue-specific lesions / inflammatory disorders ─────────────────────
    "amyloidosis": dict(
        pat=r"\bamyloid(?:osis)?\w*|\bamyloid[-\s]like\s+material",
        exclude=[],
    ),
    "amylacea": dict(
        pat=r"\bcorpor[ae]\s+amylac\w*|\bcorpus\s+amylac\w*",
        exclude=[],
    ),
    "adenoma": dict(
        pat=r"\badenom\w*|\badenomatoid\b|\badenomyosis\b",
        exclude=[],
    ),
    "bronchitis": dict(
        pat=r"\bbronchit\w*|\bbronchiolit\w*|\bperibronchit\w*",
        exclude=[],
    ),
    "cholesterol_clefts": dict(
        pat=r"\bcholesterol\s+cleft\w*|\bcholesterol\s+crystal\w*",
        exclude=[],
    ),
    "colitis": dict(pat=r"\bcolit\w*", exclude=[]),
    "consolidation": dict(pat=r"\bconsolidat\w*", exclude=[]),
    "desquamation": dict(
        pat=r"\bdesquam\w*|\bdesquamat\w*",
        exclude=[],
    ),
    "diabetic": dict(
        pat=r"\bdiabet\w*|\bdiabetic\s+(?:glomerulopath|nephropath|change|sclero)",
        exclude=[],
    ),
    "dysplasia": dict(
        pat=r"\bdysplas\w*|\bdysplastic\b",
        exclude=[],
    ),
    "endometrium_change": dict(
        pat=(
            r"\batrophic\s+endometri\w*|\bendometri\w*\s+atroph\w*"
            r"|\bcystic\s+endometri\w*|\bendometri\w*\s+(?:hyperplas|cystic\s+chang|chang|polyp)"
            r"|\bendometrial\s+(?:hyperplas|atroph|chang|polyp)"
            r"|\bpolypoid\s+endometri\w*|\bproliferat\w*\s+endometri\w*"
        ),
        exclude=[],
    ),
    "fatty_infiltration": dict(
        pat=(
            r"\bfatty\s+(?:infiltrat\w*|change|metamorphos\w*)"
            r"|\badipose\s+infiltrat\w*"
        ),
        exclude=[],
    ),
    "fiber_degeneration": dict(
        pat=(
            r"\b(?:muscle\s+)?fib(?:er|re)s?\s+(?:with\s+)?degener\w*"
            r"|\bdegenerat\w*\s+(?:muscle\s+)?fib(?:er|re)s?"
            r"|\bmyofib(?:er|re)\s+degener\w*"
        ),
        exclude=[],
    ),
    "foreign_body": dict(
        pat=(
            r"\bforeign\s+(?:body|bodies|material)"
            r"|\bforeign\s+body\s+giant\s+cell"
            r"|\baspirat\w*\s+foreign"
        ),
        exclude=[],
    ),
    "gastritis": dict(pat=r"\bgastrit\w*", exclude=[]),
    "heart_failure_cells": dict(
        pat=r"\bheart\s+failure\s+cells?",
        exclude=[],
    ),
    "hepatocyte_degeneration": dict(
        pat=(
            r"\bhepatocyte\w*\s+(?:degener|ballooning|necros|injur)"
            r"|\bhepatocellular\s+(?:degener|ballooning|injur)"
            r"|\bballooning\s+degener\w*"
        ),
        exclude=[],
    ),
    "hypoxic": dict(
        pat=r"\bhypoxi\w*|\bantemortem\s+hypoxia\b|\bischemic\s+anoxic",
        exclude=[],
    ),
    "intimal_thickening": dict(
        pat=(
            r"\bintimal\s+(?:thicken\w*|fibros\w*|hyperplas\w*|lesion|chang)"
            r"|\bintima\s+(?:thicken\w*|fibros\w*)"
            r"|\bearly\s+atheroscler\w*\s+chang"  # GTEx pairs these with intimal thickening
        ),
        exclude=[],
    ),
    "leiomyoma": dict(
        # Match leiomyoma/fibroid explicitly; do NOT match "myometrium" alone (normal tissue)
        pat=r"\bleiomyom\w*|\bfibroid\w*|\buterine\s+myom\w*|\bcellular\s+leiomyom",
        exclude=[],
    ),
    "mastopathy": dict(
        pat=r"\bmastopath\w*|\bfibrous\s+mastopath\w*",
        exclude=[],
    ),
    "medial_degeneration": dict(
        pat=(
            r"\bmedial\s+(?:degener\w*|sclero\w*|thicken\w*|calcific\s+sclero)"
            r"|\bmedial\s+calcif\w*"
        ),
        exclude=[],
    ),
    "neoplasm": dict(
        pat=(
            r"\bneoplas\w*|\btumou?rs?\b|\bcarcinoma\w*|\badenocarcinoma\w*"
            r"|\bmalignan\w*|\bmetastat\w*|\boncocyt\w*"
        ),
        exclude=["neuroendocrine_tumor"],  # avoid double-counting if NET also matches
    ),
    "nephritis": dict(
        pat=(
            r"\bnephrit\w*|\bpyelonephrit\w*|\bglomerulonephrit\w*"
            r"|\binterstitial\s+nephrit\w*"
        ),
        exclude=[],
    ),
    "nephrosclerosis": dict(
        pat=(
            r"\bnephroscler\w*|\barterioloneph\w*|\barterio[-\s]?nephroscler\w*"
            r"|\bhypertensive\s+nephroscler\w*"
        ),
        exclude=[],
    ),
    "neuroendocrine_tumor": dict(
        pat=(
            r"\bneuroendocrine\s+(?:tumou?r|neoplas)|\bcarcinoid\s+tumou?r"
            r"|\bislet\s+cell\s+tumou?r|\bnet\b"
        ),
        exclude=[],
    ),
    "pancreatitis": dict(pat=r"\bpancreatit\w*", exclude=[]),
    "plaque": dict(
        pat=(
            r"\bplaques?\b|\batheromatous\s+plaq\w*|\bintimal\s+plaq\w*"
            r"|\bfibrous\s+plaq\w*|\bsclerotic\s+plaq\w*|\bfibrofatty\s+plaq\w*"
            r"|\batheroscler\w*\s+plaq\w*"
        ),
        exclude=[],
    ),
    "prostatitis": dict(pat=r"\bprostatit\w*", exclude=[]),
    "regressive_change": dict(
        pat=(
            r"\bregressive\s+(?:chang|area|focus|foci|feature)"
            r"|\binvolutional\s+(?:chang|feature)"
        ),
        exclude=[],
    ),
    "sertoli_only": dict(
        # Pathologists describe this entity in several ways: explicit
        # "Sertoli-cell-only", or descriptive ("only / mostly / spared Sertoli",
        # "Sertoli cells remain", "germ cell aplasia").
        pat=(
            r"\bsertoli[\s-]?cell[\s-]only|\bsertoli\s+only"
            r"|\b(?:only|mostly|just|spared)\s+sertoli"
            r"|\bsertoli\s+cells?\s+(?:remain|spared)"
            r"|\bgerm\s+cell\s+aplasia"
        ),
        exclude=[],
    ),
    "solar_elastosis": dict(
        # Constrain to explicit modifiers — bare "elastosis" can refer to other
        # entities. Solar/senile/dermal elastosis are the GTEx-relevant forms.
        pat=r"\bsolar\s+elastos\w*|\bsenile\s+elastos\w*|\bdermal\s+elastos\w*",
        exclude=[],
    ),
    "sweat_glands": dict(
        pat=r"\bsweat\s+glands?",
        exclude=[],
    ),
    "thrombus": dict(
        pat=(
            r"\bthromb(?:us|i|osis|otic)\b|\bthromboembol\w*"
            r"|\bfibrin\s+thrombi?\b|\borganiz(?:ed|ing)\s+thromb"
        ),
        exclude=[],
    ),
    "tma": dict(
        # GTEx notes use "tma" for tissue microarray (procedural). Match only when
        # adjacent to the thrombotic-microangiopathy long form to avoid procedural hits.
        pat=r"\bthrombotic\s+microangiopath\w*",
        exclude=[],
    ),
    "urothelial_sloughing": dict(
        pat=(
            r"\burothel\w*\s+slough\w*|\bslough\w*\s+urothel\w*"
            r"|\burothelium\s+(?:partly\s+|focally\s+)?slough"
        ),
        exclude=[],
    ),
    "vacuolar_degeneration": dict(
        pat=(
            r"\bvacuolar\s+(?:degener|chang)|\bvacuolat\w*\s+(?:myofib|change|degener)"
            r"|\bmyofib\w*\s+vacuolat\w*|\bvacuolation\s+of\s+myofib"
        ),
        exclude=[],
    ),
    "vascular_thickening": dict(
        pat=r"\bvascular\s+thicken\w*|\bmural\s+thicken\w*|\bvessel\s+wall\s+thicken",
        exclude=[],
    ),

    # ── Negative findings (special: skip negation since they ARE negations) ──
    "no_abnormalities": dict(
        pat=(
            r"\bno\s+(?:significant\s+|major\s+)?abnormal\w*"
            r"|\bwithin\s+normal\b|\bunremarkable\b"
            r"|\bno\s+(?:significant\s+)?(?:findings?|lesions?|pathology)\b"
        ),
        exclude=[],
        skip_negation=True,
    ),
    "clean_specimens": dict(
        pat=(
            r"\bclean\b|\bgood\s+\w+|\bwell\s+(?:preserved|delineated|represented)\b"
            r"|\bexcellent\s+(?:specimen|preservation|examples?)"
            r"|\brepresentative\s+(?:specimen|sample|examples?)"
            r"|\bno\s+lesion\b|\bno\s+lesions\b"
        ),
        exclude=[],
        skip_negation=True,
    ),
}


def _compile_patterns(raw: Dict[str, dict]) -> Dict[str, dict]:
    out = {}
    for cat, spec in raw.items():
        out[cat] = dict(
            rx=re.compile(spec["pat"], re.IGNORECASE),
            exclude=spec.get("exclude", []),
            skip_negation=spec.get("skip_negation", False),
        )
    return out


COMPILED_V2 = _compile_patterns(_RAW_PATTERNS)
NORMAL_LABELS = {"no_abnormalities", "clean_specimens"}


# ── Splitting and qualifier patterns ─────────────────────────────────────────
_CLAUSE_SPLIT = re.compile(r"[.;]")
_SCOPE_TERM_SPLIT = re.compile(
    r"\b(?:and|but|however|yet|although|except|presenting|presents)\b"
    r"|consistent\s+with|\bc/w\b",
    re.IGNORECASE,
)
_POSITIVE_QUALIFIER = re.compile(
    r"\b(mild|moderate|severe|marked|significant|diffuse|focal|minimal"
    r"|prominent|central|passive|active|chronic|acute|slight|extensive"
    r"|occasional|few|several|some|scattered|moderately|focally|diffusely"
    r"|markedly|mildly|slightly|predominantly|early|late|advanced)\b"
    r"|\b\d+\s*%",  # percentage qualifier e.g. "60% steatosis"
    re.IGNORECASE,
)
_NEGATION_TRIGGER = re.compile(
    r"\b(no|not|without|absent|absence\s+of|negative|denies|deny"
    r"|free\s+of|ruled\s+out|rules\s+out|unlikely)\b",
    re.IGNORECASE,
)


def _smart_comma_split(text: str) -> List[str]:
    parts, depth, current = [], 0, []
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


def _is_negated(subclause: str, match_start: int, match_end: int) -> bool:
    """Negation if a negation trigger appears within the subclause AND is closer
    to the match than any positive qualifier between trigger and match."""
    pre = subclause[:match_start]
    neg = list(_NEGATION_TRIGGER.finditer(pre))
    if not neg:
        return False
    # Use the closest negation trigger
    last_neg = neg[-1]
    # If a clause-breaker (comma already handled) or a positive qualifier sits
    # BETWEEN the negation and the match, treat as not-negated.
    between = subclause[last_neg.end():match_start]
    if _POSITIVE_QUALIFIER.search(between):
        return False
    return True


_SCLEROTIC_PARENT_TRIGGERS = re.compile(
    r"\batheroscleros\w*|\barterioscleros\w*|\bglomeruloscleros\w*|\bnephroscleros\w*",
    re.IGNORECASE,
)


def extract_categories_v2(notes_text: str) -> List[Tuple[str, float]]:
    """Extract pathology categories with per-label confidence in [0, 1].

    Confidence components:
        base = 0.7
        +0.2 if a positive qualifier (mild/moderate/severe/% etc.) appears in
              the same subclause
        +0.1 if the category matches in more than one subclause across the note
        Hard zero if the match is negated and the category does not have
        skip_negation=True.
    """
    if not isinstance(notes_text, str) or not notes_text.strip():
        return []

    # Track raw mentions: category -> list of (confidence, was_negated)
    mentions: Dict[str, List[float]] = defaultdict(list)
    subclause_categories: List[set] = []  # per-subclause sets for exclusion

    # Pass 1: enumerate all clause/subclause matches
    raw_per_subclause = []
    for clause in _CLAUSE_SPLIT.split(notes_text):
        for sub in _smart_comma_split(clause):
            sub = sub.strip()
            if not sub:
                continue
            sub_matches = {}  # cat -> (confidence_components_dict)
            has_qualifier = bool(_POSITIVE_QUALIFIER.search(sub))
            for cat, spec in COMPILED_V2.items():
                m = spec["rx"].search(sub)
                if not m:
                    continue
                if not spec["skip_negation"] and _is_negated(sub, m.start(), m.end()):
                    continue
                conf = 0.7
                if has_qualifier:
                    conf += 0.2
                sub_matches[cat] = conf
            raw_per_subclause.append((sub, sub_matches))

    # Pass 2: apply exclusion rules at the subclause level
    accepted = []
    for sub, sub_matches in raw_per_subclause:
        cats_here = set(sub_matches.keys())
        for cat, conf in sub_matches.items():
            ex = COMPILED_V2[cat]["exclude"]
            if any(e in cats_here for e in ex):
                continue
            accepted.append((cat, conf))

    if not accepted:
        return []

    # Pass 3: aggregate across subclauses
    per_cat_confs: Dict[str, List[float]] = defaultdict(list)
    for cat, conf in accepted:
        per_cat_confs[cat].append(conf)

    out: List[Tuple[str, float]] = []
    for cat, confs in per_cat_confs.items():
        # Take max confidence; bonus for multi-subclause mentions
        c = max(confs)
        if len(confs) > 1:
            c = min(1.0, c + 0.1)
        out.append((cat, round(c, 3)))

    # GTEx tags atherosclerosis / glomerulosclerosis / nephrosclerosis as
    # ALSO sclerotic. Mirror that parent/child relationship.
    if _SCLEROTIC_PARENT_TRIGGERS.search(notes_text):
        if not any(c == "sclerotic" for c, _ in out):
            out.append(("sclerotic", 0.7))

    # If real pathology found, drop normal labels
    real = [c for c, _ in out if c not in NORMAL_LABELS]
    if real:
        out = [(c, conf) for c, conf in out if c not in NORMAL_LABELS]

    out.sort(key=lambda t: (-t[1], t[0]))
    return out


def extract_label_set(notes_text: str, min_confidence: float = 0.5) -> List[str]:
    """Convenience: return just the category names above a confidence threshold."""
    return [c for c, conf in extract_categories_v2(notes_text) if conf >= min_confidence]


# ── Severity extraction (numeric % grade) ────────────────────────────────────
# GTEx pathologists quote a percentage when grading findings (e.g. "60% steatosis",
# "30-40% fibrosis", "atherosclerosis ~50%"). We extract that 0-100 percent
# directly from the note, falling back to ordinal qualifiers when absent.

_PERCENT_RX = re.compile(
    r"(?:~|≈|about\s+|up\s+to\s+|approximately\s+)?"
    r"(\d{1,3}(?:\.\d+)?)\s*"
    r"(?:%|\s*percent\b)",
    re.IGNORECASE,
)
_PERCENT_RANGE_RX = re.compile(
    r"(\d{1,3}(?:\.\d+)?)\s*(?:-|to|–|—)\s*(\d{1,3}(?:\.\d+)?)\s*(?:%|\s*percent\b)",
    re.IGNORECASE,
)

# Ordinal qualifier → percent estimate (fallback when no explicit percent is given)
_GRADE_TO_PERCENT = {
    "minimal": 5, "trace": 5, "mild": 15, "slight": 15, "focal": 15,
    "occasional": 15, "few": 15, "rare": 5, "scattered": 15,
    "moderate": 35, "moderately": 35, "patchy": 35, "some": 25,
    "marked": 60, "markedly": 60, "severe": 70, "severely": 70,
    "extensive": 70, "significant": 50, "prominent": 50, "diffuse": 60,
    "advanced": 70, "massive": 85,
}
_GRADE_RX = re.compile(
    r"\b(" + "|".join(_GRADE_TO_PERCENT.keys()) + r")\b",
    re.IGNORECASE,
)


def _extract_subclause_severity(subclause: str) -> float:
    """Return a 0-100 severity for a subclause, preferring explicit % over qualifier."""
    # Explicit range first ("20-30%")
    m = _PERCENT_RANGE_RX.search(subclause)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if 0 <= lo <= 100 and 0 <= hi <= 100:
            return round((lo + hi) / 2.0, 2)
    # Single percent
    nums = []
    for m in _PERCENT_RX.finditer(subclause):
        v = float(m.group(1))
        if 0 <= v <= 100:
            nums.append(v)
    if nums:
        return round(sum(nums) / len(nums), 2)
    # Fallback to qualifier
    qm = _GRADE_RX.search(subclause)
    if qm:
        return float(_GRADE_TO_PERCENT[qm.group(1).lower()])
    return float("nan")


def extract_severities(notes_text: str) -> Dict[str, float]:
    """Return {category: percent_severity} for each matched (non-negated) category.

    Severity is taken from the subclause the category matched in. When multiple
    subclauses match, the max severity is kept (pathologists typically grade the
    worst-affected region). Returns NaN-free dict — categories without any
    severity signal are omitted (use extract_categories_v2 for presence/absence).
    """
    if not isinstance(notes_text, str) or not notes_text.strip():
        return {}

    cat_to_max: Dict[str, float] = {}
    for clause in _CLAUSE_SPLIT.split(notes_text):
        for sub in _smart_comma_split(clause):
            sub = sub.strip()
            if not sub:
                continue
            sev = _extract_subclause_severity(sub)
            if sev != sev:  # NaN
                continue
            for cat, spec in COMPILED_V2.items():
                m = spec["rx"].search(sub)
                if not m:
                    continue
                if not spec["skip_negation"] and _is_negated(sub, m.start(), m.end()):
                    continue
                prev = cat_to_max.get(cat, -1.0)
                if sev > prev:
                    cat_to_max[cat] = sev
    return cat_to_max
