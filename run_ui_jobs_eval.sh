#!/usr/bin/env bash
# Evaluate UI-UX Inference jobs: base vs finetuned run_final.mp4
# Jobs: 5e106411bce3, c9145455d460
#
# All videos run in ONE pipeline invocation so results are not overwritten.
#
# Usage:
#   cd /local-workspace/gowtham.s/lip-sync_inference/Evalution
#   CUDA_VISIBLE_DEVICES=0 bash run_ui_jobs_eval.sh
#   bash run_ui_jobs_eval.sh --layer1-only

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

JOBS_ROOT="/local-workspace/gowtham.s/lip-sync_inference/Finetuning-scripts/finetuning_process/UI-UX-Inference/data/jobs"
JOB_IDS=(5e106411bce3 c9145455d460)
EXTRA_ARGS=("$@")

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found on PATH" >&2
  exit 1
fi

CLI=()
for job_id in "${JOB_IDS[@]}"; do
  base="${JOBS_ROOT}/${job_id}/infer_base/run_final.mp4"
  ft="${JOBS_ROOT}/${job_id}/infer_ft/run_final.mp4"
  if [[ ! -f "$base" ]]; then
    echo "ERROR: missing base output: $base" >&2
    exit 1
  fi
  if [[ ! -f "$ft" ]]; then
    echo "ERROR: missing finetuned output: $ft" >&2
    exit 1
  fi
  CLI+=(--video "$base" --model xdub_base --video "$ft" --model xdub_ft)
done

echo "============================================================"
echo " UI jobs eval (${#JOB_IDS[@]} jobs, $(( ${#JOB_IDS[@]} * 2 )) videos)"
for job_id in "${JOB_IDS[@]}"; do
  echo "   ${job_id}: infer_base + infer_ft run_final.mp4"
done
echo "============================================================"

python run_pipeline.py "${EXTRA_ARGS[@]}" "${CLI[@]}"

echo
echo "Done. Results:"
echo "  ${ROOT}/results/metrics_raw.csv"
echo "  ${ROOT}/results/metrics_by_shot_scale.csv"
echo "  ${ROOT}/results/metrics_by_difficulty.csv"
echo "  ${ROOT}/results/qc_flags.csv"
echo "  ${ROOT}/results/metrics_tier_summary.json"
echo "  ${ROOT}/results/human_eval/"
echo "  ${ROOT}/results/segments.csv"
