#!/usr/bin/env bash
set -euo pipefail

# Run stratified fairness gaps across the three MI-clean result folders.
#
# For each folder X in {mi_race_clean, mi_age_clean, mi_gender_clean}:
#   - compute all three gaps:
#       1) race_major gap within age_group×gender strata
#       2) gender gap within age_group×race_major strata
#       3) age_group gap within race_major×gender strata
#   - run for both metrics: greenscore and radgraph_f1
#   - scale metrics to 0-100 (percentage points) by default via --metric_scale auto
#   - cache radgraph_f1 per-id so reruns are fast
#
# Outputs go next to the input folder:
#   <input_folder>/fairness_gap__strata_<...>__gap_<...>__<metric>/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="$ROOT_DIR/results"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="$ROOT_DIR/scripts/rebuttal/cross_sectional_fairness_gap.py"

MIN_GROUP_N="${MIN_GROUP_N:-20}"
WEIGHTING="${WEIGHTING:-stratum_n}"   # stratum_n | equal
METRIC_SCALE="${METRIC_SCALE:-auto}"  # auto | 100 | none

usage() {
  cat <<'EOF'
Usage:
  bash scripts/rebuttal/run_all_stratified_fairness.sh [FOLDER_OR_PATH ...]

Examples:
  # Run all three default folders
  bash scripts/rebuttal/run_all_stratified_fairness.sh

  # Run just one folder
  bash scripts/rebuttal/run_all_stratified_fairness.sh mi_age_clean

  # Run an arbitrary results folder path
  bash scripts/rebuttal/run_all_stratified_fairness.sh results/some_random_exp

  # Run an arbitrary absolute path
  bash scripts/rebuttal/run_all_stratified_fairness.sh /abs/path/to/some_random_exp

  # Or pass the JSONL directly
  bash scripts/rebuttal/run_all_stratified_fairness.sh /abs/path/to/merged_demographics.jsonl

  # Run two folders
  bash scripts/rebuttal/run_all_stratified_fairness.sh mi_age_clean mi_gender_clean

Resolution rules for each argument:
  1) If it's a file path ending in .jsonl, use it as input.
  2) Else if it's a directory, look for <dir>/merged_demographics.jsonl.
  3) Else treat it as a folder name under results/: results/<name>/merged_demographics.jsonl.

Env knobs:
  MIN_GROUP_N=20   WEIGHTING=stratum_n   METRIC_SCALE=auto   PYTHON_BIN=python
EOF
}

resolve_input() {
  # Prints: <input_jsonl> <base_out_dir> <label>
  local arg="$1"

  # Case 1: direct jsonl path
  if [[ -f "$arg" && "$arg" == *.jsonl ]]; then
    local abs
    abs="$(cd "$(dirname "$arg")" && pwd)/$(basename "$arg")"
    echo "$abs" "$(dirname "$abs")" "$(basename "$(dirname "$abs")")"
    return 0
  fi

  # Case 2: directory path
  if [[ -d "$arg" ]]; then
    local p="$arg/merged_demographics.jsonl"
    if [[ -f "$p" ]]; then
      local abs_dir
      abs_dir="$(cd "$arg" && pwd)"
      echo "$abs_dir/merged_demographics.jsonl" "$abs_dir" "$(basename "$abs_dir")"
      return 0
    fi
  fi

  # Case 3: treat as results/<name>
  local p="$RESULTS_DIR/$arg/merged_demographics.jsonl"
  if [[ -f "$p" ]]; then
    local abs_dir
    abs_dir="$(cd "$RESULTS_DIR/$arg" && pwd)"
    echo "$abs_dir/merged_demographics.jsonl" "$abs_dir" "$arg"
    return 0
  fi

  echo "" "" ""  # signal failure
  return 1
}

run_one() {
  local label="$1" base_out_dir="$2" input="$3" gap_field="$4" strata_fields="$5" radgraph_cache_csv="$6"

  # GreenScore
  local out_gs="$base_out_dir/fairness_gap__strata_${strata_fields//,/_x_}__gap_${gap_field}__greenscore"
  mkdir -p "$out_gs"
  "$PYTHON_BIN" "$SCRIPT" \
    --input "$input" \
    --metric greenscore \
    --metric_scale "$METRIC_SCALE" \
    --strata_fields "$strata_fields" \
    --gap_field "$gap_field" \
    --min_group_n "$MIN_GROUP_N" \
    --weighting "$WEIGHTING" \
    --outdir "$out_gs" \
    >/dev/null

  # RadGraph-F1 (computed if missing)
  local out_rg="$base_out_dir/fairness_gap__strata_${strata_fields//,/_x_}__gap_${gap_field}__radgraphf1"
  mkdir -p "$out_rg"
  "$PYTHON_BIN" "$SCRIPT" \
    --input "$input" \
    --metric radgraph_f1 \
    --metric_scale "$METRIC_SCALE" \
    --strata_fields "$strata_fields" \
    --gap_field "$gap_field" \
    --min_group_n "$MIN_GROUP_N" \
    --weighting "$WEIGHTING" \
    --radgraph_cache_csv "$radgraph_cache_csv" \
    --outdir "$out_rg" \
    >/dev/null

  echo "[OK] $label :: gap=${gap_field} strata=${strata_fields} (MIN_GROUP_N=$MIN_GROUP_N, WEIGHTING=$WEIGHTING, METRIC_SCALE=$METRIC_SCALE)"
  echo "     - $out_gs"
  echo "     - $out_rg"
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi
  if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: fairness script not found at: $SCRIPT" >&2
    exit 1
  fi

  # Analyses to run per input. Each JSONL is expected to contain age_group, gender, race_major.
  # Note: age gaps are computed over the categorical bins in 'age_group'.
  local args=()
  if [[ "$#" -gt 0 ]]; then
    args=("$@")
  else
    args=("mi_race_clean" "mi_age_clean" "mi_gender_clean")
  fi

  for arg in "${args[@]}"; do
    local input base_out_dir label
    if ! read -r input base_out_dir label < <(resolve_input "$arg"); then
      echo "WARN: could not resolve input from: $arg" >&2
      continue
    fi
    if [[ -z "$input" || -z "$base_out_dir" ]]; then
      echo "WARN: could not resolve input from: $arg" >&2
      continue
    fi

    # Reuse one cache per input folder (radgraph_f1 is per-id and independent of stratification).
    local cache_dir="$base_out_dir/fairness_gap__radgraph_cache"
    mkdir -p "$cache_dir"
    local radgraph_cache_csv="$cache_dir/radgraph_cache.csv"

    # 1) race_major gap within age_group×gender strata
    run_one "$label" "$base_out_dir" "$input" "race_major" "age_group,gender" "$radgraph_cache_csv"

    # 2) gender gap within age_group×race_major strata (keep age+race constant)
    run_one "$label" "$base_out_dir" "$input" "gender" "age_group,race_major" "$radgraph_cache_csv"

    # 3) age_group gap within race_major×gender strata
    run_one "$label" "$base_out_dir" "$input" "age_group" "race_major,gender" "$radgraph_cache_csv"
  done

  echo
  echo "Done. Summaries are in each output dir as summary.json"
}

main "$@"
