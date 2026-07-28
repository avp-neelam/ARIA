#!/bin/bash
#SBATCH -J aria-ensemble
#SBATCH -p coskunuzer #coskunuzer # swap to -p coskunuzer if the lab's condo GPU nodes are free (check: sinfo -p coskunuzer) -- not subject to preemption
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 23:00:00
#SBATCH -o /groups/bcoskunuzer/avpn/ARIA/aria_benchmark/logs/%x-%j.out
#SBATCH --mail-user=neelam.akula@utdallas.edu
#SBATCH --mail-type=END,FAIL

# Phase 3/4: assembles + evaluates ARIA-Ensemble (aria_ph + aria_phg +
# aria_moe, logit-averaged) using each member's winning config from phase 2's
# results/aria_search.json. Submitted automatically by 02_aria_search.slurm.
#
# Same self-resubmit-until-done, then chain-to-next-phase pattern.

set -uo pipefail
# NOTE: can't use "$(dirname "$0")" here -- SLURM copies this script to a
# spool directory before running it, so $0 does NOT point back at the real
# slurm/ directory in the repo.
ARIA_SLURM_DIR="/groups/bcoskunuzer/avpn/ARIA/aria_benchmark/slurm"
source "${ARIA_SLURM_DIR}/common.sh"

TIME_BUDGET=79200   # ~22h
LOG="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"

"$PYBIN" scripts/run_ensemble_eval.py \
    --time_budget "$TIME_BUDGET" \
    --state_dir state/ensemble_eval \
    --results_file results/ensemble_eval.json \
    --aria_results results/aria_search.json \
    --device auto \
    --save_artifacts --artifacts_dir artifacts

if grep -q "ALL_DONE" "$LOG"; then
    echo "Ensemble eval complete -- submitting final aggregation (phase 4)."
    sbatch "${ARIA_SLURM_DIR}/04_aggregate.slurm"
else
    echo "Not finished yet -- resubmitting ensemble eval."
    sbatch "${ARIA_SLURM_DIR}/03_ensemble_eval.slurm"
fi
