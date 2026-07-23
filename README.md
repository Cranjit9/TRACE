# TRACE

TRACE (Transcriptomic Reading of Affected organs and Causal Evaluation) is a confounder-aware framework asking two sequential questions across a pathology-linked GTEx benchmark: which organ pathologies leave a detectable trace in the whole-blood transcriptome beyond donor demographics and procedural covariates, and whether that trace is causal or reactive. Each pathology is first tested against a clinical-covariate baseline using per-fold covariate orthogonalization on principal-component features, and surviving signals are triaged through cis-pQTL-anchored Mendelian randomization against UK Biobank plasma proteomics and FinnGen R12 outcomes. Only a minority of pathologies clear the detection gate, with liver-cirrhosis emerging as the strongest signal and SERPINE1 as its leading causal driver.

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
| 16 | Main manuscript figures 1–4 |
| 17 | Supplementary figures S1–S6 |

Parameters and thresholds are consolidated in `gtex_biomarkers/config.py`.

## Layout

```
gtex_biomarkers/  # config, data, models, evaluation, figure builders
notebooks/        # NB01-NB17
scripts/          # pipeline runner + figure composer
manuscript/       # local drafts (gitignored)
data/             # inputs (gitignored)
output/           # results (gitignored)
```

## License

Apache 2.0 — see [LICENSE](LICENSE). Copyright 2025 Sanju Sinha, Sanford Burnham Prebys Medical Discovery Institute.
