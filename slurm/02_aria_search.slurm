#!/bin/bash
#SBATCH -J aria-search
#SBATCH -p coskunuzer #coskunuzer # swap to -p coskunuzer if the lab's condo GPU nodes are free (check: sinfo -p coskunuzer) -- not subject to preemption
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 23:00:00
#SBATCH -o /groups/bcoskunuzer/avpn/ARIA/aria_benchmark/logs/%x-%j.out
#SBATCH --mail-user=neelam.akula@utdallas.edu
#SBATCH --mail-type=END,FAIL

# Phase 2/4: hyperparameter search + final evaluation for the ARIA family
# (aria, aria_ph, aria_phg, aria_moe) across all 13 datasets. Submitted
# automatically by 01_baseline_search.slurm once phase 1 finishes -- you
# shouldn't need to sbatch this one by hand unless resuming manually.
#
# Same self-resubmit-until-done, then chain-to-next-phase pattern as phase 1.
# This is the biggest grid in the whole sweep (aria_ph/aria_phg are 768
# configs each) -- expect this phase to take the most wall-clock time and
# the most resubmissions.

set -uo pipefail
# NOTE: can't use "$(dirname "$0")" here -- SLURM copies this script to a
# spool directory before running it, so $0 does NOT point back at the real
# slurm/ directory in the repo.
ARIA_SLURM_DIR="/groups/bcoskunuzer/avpn/ARIA/aria_benchmark/slurm"
source "${ARIA_SLURM_DIR}/common.sh"

TIME_BUDGET=79200   # ~22h
LOG="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"

"$PYBIN" scripts/run_aria_search.py \
    --time_budget "$TIME_BUDGET" \
    --state_dir state/aria_search \
    --results_file results/aria_search.json \
    --device auto \
    --save_artifacts --save_weights --artifacts_dir artifacts

if grep -q "ALL_DONE" "$LOG"; then
    echo "ARIA search complete -- submitting ensemble eval (phase 3) next."
    sbatch "${ARIA_SLURM_DIR}/03_ensemble_eval.slurm"
else
    echo "Not finished yet -- resubmitting ARIA search."
    sbatch "${ARIA_SLURM_DIR}/02_aria_search.slurm"
fi
