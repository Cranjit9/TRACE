# TRACE

Whole-blood gene expression as a non-invasive proxy for organ-specific tissue pathology. Reproduces the analyses and figures for the manuscript.

**Manuscript**: `manuscript/manuscript_v5_TRACE.docx` and `manuscript/supplementary_v5_TRACE.docx`.

## Setup

Python 3.12.8, pinned via `requirements-lock.txt`.

```bash
git clone https://github.com/Cranjit9/gtex_gene_expression.git
cd gtex_gene_expression
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip wheel setuptools
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e .
```

## Data

Place the following GTEx v10 files in `data/raw/`:

| File | Source |
|---|---|
| `GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct` | GTEx Portal |
| `GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt` | GTEx Portal |
| `Gtex_restricted.txt` | dbGaP restricted access |
| `meta_data_with_url.csv` | GTEx Portal |

## Run

```bash
./scripts/run_all_nbs.sh   # end-to-end, ~4 h
```

## Notebooks

| NB | Purpose |
|---|---|
| 01–03 | GTEx load, PCA, NLP imputation of pathology labels |
| 04–09 | Binary + multi-category classifiers, LR/RF baselines, confounder analysis |
| 10–12 | PC + covariate screen, gene back-projection, pathway enrichment |
| 13–15 | Cross-organ plasma enrichment, UKB-PPP Mendelian randomization, liver tissue check |
| 16 | Early multipanel figures (pre-approval) |
| 17 | Main manuscript figures 1–4 |
| 18 | Supplementary figures S1–S6 + `supplementary.docx` |

Parameters and thresholds are consolidated in `gtex_biomarkers/config.py`.

## Layout

```
gtex_biomarkers/  # config, data, models, evaluation, figure builders
notebooks/        # NB01-NB18
scripts/          # pipeline runner + figure composer
manuscript/       # local drafts (gitignored)
data/             # inputs (gitignored)
output/           # results (gitignored)
```

## License

Apache 2.0 — see [LICENSE](LICENSE). Copyright 2025 Sanju Sinha, Sanford Burnham Prebys Medical Discovery Institute.
