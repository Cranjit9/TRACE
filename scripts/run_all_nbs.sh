#!/usr/bin/env bash
set -euo pipefail

# Execute the full pipeline (NB01-NB17) into output/executed_notebooks/;
# per-notebook logs land in _runlogs/. Source notebooks are not modified.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NB_DIR="${ROOT}/notebooks"
EXEC_DIR="${ROOT}/output/executed_notebooks"
LOG_DIR="${ROOT}/_runlogs"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
JUPYTER="${JUPYTER:-${ROOT}/.venv/bin/jupyter}"

cd "${ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python interpreter not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -x "${JUPYTER}" ]]; then
  echo "Jupyter executable not found: ${JUPYTER}" >&2
  exit 1
fi

mkdir -p "${EXEC_DIR}" "${LOG_DIR}"

NOTEBOOKS=(
  01_data_loading_exploration
  02_liver_steatosis_binary
  03_nlp_imputation
  04_liver_multicategory
  05_all_tissue_models
  06_summary_baseline
  07_rf_variance_filter
  08_comparison
  09_confounder_analysis
  10_tissue_pc_regression
  11_pc_gene_importance
  12_pathway_enrichment
  13_organ_enrichment
  14_ukbpp_validation
  15_liver_tissue_validation
  16_main_figures
  17_supplementary
)

for stem in "${NOTEBOOKS[@]}"; do
  input="${NB_DIR}/${stem}.ipynb"
  if [[ ! -f "${input}" ]]; then
    echo "Missing canonical notebook: ${input}" >&2
    exit 1
  fi
  echo "Executing ${stem}"
  "${JUPYTER}" nbconvert --to notebook --execute "${input}" \
    --output "${stem}.executed.ipynb" \
    --output-dir "${EXEC_DIR}" \
    --ExecutePreprocessor.timeout=10800 \
    >"${LOG_DIR}/${stem}.log" 2>&1
done

echo "All notebooks executed successfully."
