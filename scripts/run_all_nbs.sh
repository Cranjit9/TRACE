#!/usr/bin/env bash
# Run NB01 → NB18 from a clean state with memory cleanup between each NB.
#
# Memory protocol:
#   - Asks for sudo once at start (so `purge` can be called between NBs).
#   - A background loop refreshes the sudo grant every 60s so it doesn't expire.
#   - After each NB completes, runs `sudo purge` to flush inactive memory caches,
#     then logs `vm_stat` and swap usage so we can spot pressure building up.
#
# Output:
#   _runlogs/STATUS.txt          — high-level per-NB pass/fail + timing + memory
#   _runlogs/<NB>.log            — per-NB nbconvert output (errors here on failure)

set -u

NB_DIR="/Users/rsinha/Library/CloudStorage/OneDrive-SanfordBurnhamPrebysMedicalDiscoveryInstitute/Desktop/gtex_gene_expression/notebooks"
LOG_DIR="$NB_DIR/_runlogs"
STATUS="$LOG_DIR/STATUS.txt"
JUPYTER="/Users/rsinha/Library/CloudStorage/OneDrive-SanfordBurnhamPrebysMedicalDiscoveryInstitute/Desktop/gtex_gene_expression/.venv/bin/jupyter"

mkdir -p "$LOG_DIR"
: > "$STATUS"
cd "$NB_DIR"

# ── Grab sudo upfront so purge can run between NBs without prompting again ──
echo "Need sudo so 'purge' can flush memory between NB runs."
sudo -v || { echo "sudo failed — aborting"; exit 1; }

# Keep sudo grant alive in the background for the whole run
( while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done ) &
SUDO_KEEPER=$!
trap 'kill $SUDO_KEEPER 2>/dev/null' EXIT

# ── Helpers ─────────────────────────────────────────────────────────────────
mem_line() {
    # Compact one-liner: free / inactive / swap
    local free=$(vm_stat | awk '/Pages free/{gsub(/\./,"",$3); printf "%.0f", $3*16/1024}')
    local inact=$(vm_stat | awk '/Pages inactive/{gsub(/\./,"",$3); printf "%.0f", $3*16/1024}')
    local swap=$(sysctl -n vm.swapusage | awk '{print $6}')
    echo "free=${free}MB inactive=${inact}MB swap_used=${swap}"
}

# ── Notebook list ───────────────────────────────────────────────────────────
NBS=(
  "01_data_loading_exploration"
  "02_liver_steatosis_binary"
  "03_nlp_imputation"
  "04_liver_multicategory"
  "05_all_tissue_models"
  "06_summary_baseline"
  "07_rf_variance_filter"
  "08_comparison"
  "09_confounder_analysis"
  "10_tissue_pc_regression"
  "11_pc_gene_importance"
  "12_pathway_enrichment"
  "13_organ_enrichment"
  "14_ukbpp_validation"
  "15_liver_tissue_validation"
  "16_multipanel_figures"
  "17_supp_methodology"
  "18_supplementary_doc"
)

OVERALL_START=$(date +%s)
echo "Run started: $(date)"                  >> "$STATUS"
echo "Notebooks: ${#NBS[@]}"                 >> "$STATUS"
echo "Initial memory: $(mem_line)"           >> "$STATUS"
echo ""                                       >> "$STATUS"

# ── Main loop ───────────────────────────────────────────────────────────────
for nb in "${NBS[@]}"; do
    log="$LOG_DIR/${nb}.log"
    start=$(date +%s)
    mem_pre=$(mem_line)
    echo "[$(date '+%H:%M:%S')] START $nb  | $mem_pre" | tee -a "$STATUS"

    "$JUPYTER" nbconvert --to notebook --execute "${nb}.ipynb" --inplace \
        --ExecutePreprocessor.timeout=10800 > "$log" 2>&1
    rc=$?

    end=$(date +%s)
    dur=$((end - start))
    dur_min=$((dur / 60))
    dur_sec=$((dur % 60))

    if [ $rc -ne 0 ]; then
        mem_post=$(mem_line)
        echo "[$(date '+%H:%M:%S')] FAIL  $nb  (${dur_min}m ${dur_sec}s)  rc=$rc  | $mem_post" | tee -a "$STATUS"
        echo ""                                       >> "$STATUS"
        echo "Tail of failure log:"                    >> "$STATUS"
        tail -40 "$log"                                 >> "$STATUS"
        exit $rc
    fi

    # Memory cleanup — give Python time to exit, then purge inactive caches
    sleep 3
    sudo -n purge 2>/dev/null
    sleep 2
    mem_post=$(mem_line)

    echo "[$(date '+%H:%M:%S')] OK    $nb  (${dur_min}m ${dur_sec}s)  | $mem_post" | tee -a "$STATUS"
done

OVERALL_END=$(date +%s)
TOTAL=$((OVERALL_END - OVERALL_START))
echo ""                                                       >> "$STATUS"
echo "All notebooks OK. Total: $((TOTAL / 60))m $((TOTAL % 60))s" | tee -a "$STATUS"
echo "Final memory: $(mem_line)"                              >> "$STATUS"
