#!/bin/bash
#SBATCH -J aria-aggregate
#SBATCH -p cpu-preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o /groups/bcoskunuzer/avpn/ARIA/aria_benchmark/logs/%x-%j.out
#SBATCH --mail-user=neelam.akula@utdallas.edu
#SBATCH --mail-type=END,FAIL

# Phase 4/4 (final step, no chaining after this): combines
# baseline_search.json + aria_search.json + ensemble_eval.json into one
# markdown results table. CPU-only, seconds to run -- `cpu-preempt` is the
# right fit here, no need to touch the GPU queue for this. Submitted
# automatically by 03_ensemble_eval.slurm.

set -uo pipefail
# NOTE: can't use "$(dirname "$0")" here -- SLURM copies this script to a
# spool directory before running it, so $0 does NOT point back at the real
# slurm/ directory in the repo.
ARIA_SLURM_DIR="/groups/bcoskunuzer/avpn/ARIA/aria_benchmark/slurm"
source "${ARIA_SLURM_DIR}/common.sh"

"$PYBIN" scripts/aggregate_results.py --out results/FINAL_RESULTS.md
echo "Done. Final table: results/FINAL_RESULTS.md"
