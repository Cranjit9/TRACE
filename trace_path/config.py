"""Centralised configuration for the GTEx blood-based biomarkers pipeline."""

from pathlib import Path

class Config:
    """All tuneable parameters in one place."""

    ROOT_DIR = Path(__file__).resolve().parent.parent
    RAW_DIR = ROOT_DIR / "data" / "raw"
    CACHE_DIR = ROOT_DIR / "data" / "cache"
    PROCESSED_DIR = ROOT_DIR / "data" / "processed"
    TABLES_DIR = ROOT_DIR / "output" / "tables"
    FIGURES_DIR = ROOT_DIR / "output" / "figures"

    EXPR_FILE = RAW_DIR / "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct"
    META_FILE = RAW_DIR / "GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt"
    AGE_FILE = RAW_DIR / "Gtex_restricted.txt"
    PATHOLOGY_FILE = RAW_DIR / "meta_data_with_url.csv"

    SEED = 0
    N_SPLITS = 5

    TOP_K_FEATURES = 100
    N_TOP_VAR_GENES = 20_000

    SAMPLE_THRESHOLD_LIVER = 5
    ALL_TISSUE_THRESHOLD = 50
    MIN_POS_NEG_BLOOD = 5

    LR_SOLVER = "saga"
    LR_MAX_ITER = 5_000
    RF_N_ESTIMATORS = 500
    RF_MAX_FEATURES = "sqrt"

    AUC_THRESH = 0.60      # qualifying gate: PC+co-variates expression AUC
    DELTA_THRESH = 0.05    # qualifying gate: delta (expr gain over clinical co-variates)

    COVARIATE_ORTHO_AUC_THRESH = 0.70             # headline (per-PC max-AUC drop cutoff)
    COVARIATE_ORTHO_THRESHOLDS = [0.65, 0.70, 0.75]  # sweep set for sensitivity analysis

    NORMAL_LABELS = {"clean_specimens", "no_abnormalities"}

    @classmethod
    def ensure_dirs(cls):
        """Create output directories if they don't exist."""
        for d in [cls.RAW_DIR, cls.CACHE_DIR, cls.PROCESSED_DIR, cls.TABLES_DIR, cls.FIGURES_DIR]:
            d.mkdir(parents=True, exist_ok=True)
