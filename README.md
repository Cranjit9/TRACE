# Blood-Based Biomarkers for Tissue Pathology (GTEx)

Can **Whole Blood gene expression** serve as a non-invasive proxy to detect **organ-specific tissue pathology**? This project uses GTEx v10 matched multi-tissue and blood expression data to build and evaluate blood-based classifiers for pathology conditions across the full GTEx tissue panel.

## Pipeline overview

The pipeline screens every accepted tissue × pathology pair through a consistent leak-free 5-fold grouped cross-validation, then promotes pairs that pass the configured AUC and Δ-AUC gates to gene- and pathway-level interpretation:

1. **Label assembly** — NLP imputation of free-text pathology notes augments GTEx's structured categories. The accepted pair pool is whatever passes the minimum-sample threshold defined in `Config`.
2. **Model screen** — Logistic Regression and Random Forest models are evaluated on raw and variance-filtered blood expression, with and without clinical co-variates (age, sex, race, death circumstances, ischemic time).
3. **PC + co-variate screen** — `StandardScaler → PCA → univariate-AUC PC selection → Random Forest` fit inside each CV fold. The qualifying gate is `auc_pc_conf ≥ Config.AUC_THRESH` AND `delta_pc ≥ Config.DELTA_THRESH` (see `gtex_biomarkers/config.py`).
4. **Gene back-projection** — Selected-PC importances are back-projected to genes via normalized PCA loadings to obtain per-gene importance for each qualifying pair.
5. **Pathway enrichment** — GSEA prerank on the full PC-derived gene rankings against KEGG, Reactome, and GO Biological Process.
6. **Organ validation** — GSEA against Oh et al. 2023 ST5 organ plasma proteins and FDA/clinical blood biomarkers; two-layer integrated scorecard and flagship gene list.
7. **External validation** — Mendelian randomization against UK Biobank pQTL panels and an independent liver-tissue differential-expression check.

The pair counts at every stage are derived from the data — the README does not hardcode them. Run the notebooks end-to-end for the current canonical numbers (printed in NB06 / NB10 / NB16).

## Installation

The project ships a pinned environment that reproduces published numerical results bit-for-bit. Use the `.venv` workflow below for the canonical setup; the conda path is retained for legacy users.

```bash
# Clone
git clone https://github.com/Cranjit9/gtex_gene_expression.git
cd gtex_gene_expression

# Recommended: pinned venv (Python 3.12.8)
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip wheel setuptools
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e .
.venv/bin/python -m ipykernel install --user --name gtex-venv --display-name "Python 3.12.8 (gtex .venv)"

# Legacy alternative: conda or unpinned pip
conda env create -f env.yaml && conda activate gtex_biomarkers
# or: pip install -r requirements.txt
```

`requirements-lock.txt` pins every transitive dependency (124 packages) at the versions used to generate the manuscript results. `requirements.txt` carries floors only and is kept for downstream packaging.

## Data

Download the following files from [GTEx Portal](https://gtexportal.org/) and place them in `data/raw/`:

| File | Description |
|------|-------------|
| `GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct` | Gene TPM expression matrix (all tissues) |
| `GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt` | Sample-level metadata |
| `Gtex_restricted.txt` | Donor-level restricted data (AGE, SEX) — requires dbGaP approved access; contact GTEx to request project-specific authorization |
| `meta_data_with_url.csv` | Tissue pathology categories and notes |

## Usage

Run the full pipeline (~4 hours):

```bash
./scripts/run_all_nbs.sh
```

Or open notebooks individually:

```
notebooks/
├── 01_data_loading_exploration.ipynb   # Load GTEx, filter blood, PCA
├── 02_liver_steatosis_binary.ipynb     # Binary steatosis proof of concept
├── 03_nlp_imputation.ipynb             # Impute missing pathology labels
├── 04_liver_multicategory.ipynb        # All liver pathology categories
├── 05_all_tissue_models.ipynb          # Pan-tissue LR baseline
├── 06_summary_baseline.ipynb           # Bar chart, heatmap
├── 07_rf_variance_filter.ipynb         # RF + 20K variance filter
├── 08_comparison.ipynb                 # LR vs RF
├── 09_confounder_analysis.ipynb        # Confounder-only vs expression+confounder RF
├── 10_tissue_pc_regression.ipynb       # PC screening + PC+confounder RF + six-model summary
├── 11_pc_gene_importance.ipynb         # PC importance back-projected to genes
├── 12_pathway_enrichment.ipynb         # GSEA on PC-derived rankings
├── 13_organ_enrichment.ipynb           # ST5 organ + FDA biomarkers, two-layer scorecard
├── 14_ukbpp_validation.ipynb           # UK Biobank pQTL Mendelian randomization
├── 15_liver_tissue_validation.ipynb    # Independent liver tissue cross-check
├── 16_multipanel_figures.ipynb         # Main manuscript figures
├── 17_supp_methodology.ipynb           # Supplementary methodology figures
├── 18_supplementary_doc.ipynb          # Builds `manuscript/supplementary.docx`
└── 19_all_results_dump.ipynb           # Dumps every CSV to `output/tables/all_results.txt`
```

Shared code lives in `gtex_biomarkers/` (data loading, models, evaluation, figure builders).

## Project Structure

```
gtex_gene_expression/
├── README.md
├── LICENSE                        # Apache 2.0
├── pyproject.toml                 # Build config + dep floors
├── requirements.txt               # Pip floors (downstream packaging)
├── requirements-lock.txt          # Pinned manuscript environment
├── env.yaml                       # Conda environment (legacy)
├── gtex_biomarkers/               # Python package
│   ├── config.py                  # Parameters & paths (single source of truth)
│   ├── data.py                    # Loading, blood matrix, variance filter
│   ├── labels.py                  # Donor labels, NLP imputation, pair discovery
│   ├── models.py                  # CV pipelines (LR, RF), feature selection
│   ├── evaluation.py              # ROC, PR, CM, boxplot, summary plots
│   ├── figure_builder.py          # Multipanel manuscript figures
│   └── utils.py                   # Parallel runners, comparison tables
├── notebooks/                     # Analysis notebooks (NB01-NB19)
├── scripts/
│   └── run_all_nbs.sh             # End-to-end pipeline runner
├── manuscript/                    # Main + supplementary docx
├── data/                          # Not tracked in git
│   ├── raw/                       # GTEx downloads
│   └── processed/                 # Imputed labels
└── output/                        # Not tracked in git
    ├── figures/                   # PNG plots
    └── tables/                    # CSV results
```

## Methodology

Listed in pipeline order (labels → CV → screen → PC → interpretation → calibration).

- **NLP imputation**: Regex-based category extraction with ConText-inspired negation detection augments GTEx's structured pathology categories
- **Cross-validation**: 5-fold `StratifiedGroupKFold` grouped by donor SUBJID (prevents leakage); seed and split count from `Config`
- **Feature selection**: Per-fold AUC-based ranking, `Config.TOP_K_FEATURES` (train-only)
- **Models**: Logistic Regression (baseline) and Random Forest (`Config.RF_N_ESTIMATORS` trees, balanced classes)
- **Co-variate analysis**: Age, sex, race, death circumstances, and ischemic time imputed within each train fold and compared against expression-only and expression+co-variate models
- **PC regression**: `StandardScaler → PCA → univariate AUC selection of PCs → Random Forest`, fit inside each CV fold
- **Qualifying gate**: `Config.AUC_THRESH` (expression AUC) and `Config.DELTA_THRESH` (expression gain over clinical co-variates), single source of truth, no literals elsewhere in the codebase
- **Gene attribution from PCs**: Selected PC importances back-projected to genes using normalized PCA loadings; unannotated `ENSG`-prefix entries dropped at source
- **Pathway analysis**: GSEA prerank on full PC-derived gene rankings; gseapy seeded with `Config.SEED`
- **Threshold tuning**: Youden's J statistic (maximises sensitivity + specificity)

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Copyright 2025 Sanju Sinha, Sanford Burnham Prebys Medical Discovery Institute.
