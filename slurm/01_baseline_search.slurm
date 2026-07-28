#!/bin/bash
#SBATCH -J aria-baseline
#SBATCH -p coskunuzer #coskunuzer # swap to -p coskunuzer if the lab's condo GPU nodes are free (check: sinfo -p coskunuzer) -- not subject to preemption
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 23:00:00
#SBATCH -o /groups/bcoskunuzer/avpn/ARIA/aria_benchmark/logs/%x-%j.out
#SBATCH --mail-user=neelam.akula@utdallas.edu
#SBATCH --mail-type=END,FAIL

# Phase 1/4 of the full sweep: hyperparameter search + final evaluation for
# the 5 baselines (MLP, GCN, GraphSAGE, GAT, GPS) across all 13 datasets.
#
# Self-resubmitting: this job runs for --time_budget seconds (kept under the
# --time above), then checks its own log for "ALL_DONE". If not finished, it
# resubmits itself to keep going from the checkpointed state/*.json; if
# finished, it kicks off phase 2 (ARIA search) automatically. Submit this
# ONE script once -- 02/03/04 chain on their own from here.
#
# Runs directly against /groups (not copied to node-local scratch first, the
# way an earlier project on this cluster did it): this project checkpoints
# state to disk after every single training run, specifically to survive
# gpu-preempt's instant, no-grace-period kills -- copying to scratch and
# syncing back only at job end would defeat that, since a preemption mid-job
# would lose everything since the last sync instead of just the one split in
# flight. The datasets/state/results here are small (small graphs, JSON
# checkpoints, a few hundred KB-MB per artifact file), so MooseFS's lower
# throughput vs local scratch doesn't matter much for this workload.

set -uo pipefail
# NOTE: can't use "$(dirname "$0")" here -- SLURM copies this script to a
# spool directory before running it (e.g. /var/spool/slurm/d/job.../slurm_script),
# so $0 does NOT point back at the real slurm/ directory in the repo.
ARIA_SLURM_DIR="/groups/bcoskunuzer/avpn/ARIA/aria_benchmark/slurm"
source "${ARIA_SLURM_DIR}/common.sh"

# Comfortably under --time above so the script checkpoints and exits on its
# own before SLURM would kill it. This is a soft target, not the only
# safety net -- state/*.json is saved after every single training run
# regardless, which is what actually protects against preemption.
TIME_BUDGET=79200   # ~22h

LOG="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"

"$PYBIN" scripts/run_baseline_search.py \
    --time_budget "$TIME_BUDGET" \
    --state_dir state/baseline_search \
    --results_file results/baseline_search.json \
    --device auto \
    --save_artifacts --save_weights --artifacts_dir artifacts

if grep -q "ALL_DONE" "$LOG"; then
    echo "Baseline search complete -- submitting ARIA search (phase 2) next."
    sbatch "${ARIA_SLURM_DIR}/02_aria_search.slurm"
else
    echo "Not finished yet -- resubmitting baseline search."
    sbatch "${ARIA_SLURM_DIR}/01_baseline_search.slurm"
fi
