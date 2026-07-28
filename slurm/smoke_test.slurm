#!/bin/bash
#SBATCH -J aria-smoke
#SBATCH -p cpu-preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /groups/bcoskunuzer/avpn/ARIA/aria_benchmark/logs/%x-%j.out
#SBATCH --mail-user=neelam.akula@utdallas.edu
#SBATCH --mail-type=END,FAIL

# Run this BEFORE anything else. `cpu-preempt` is a large, CPU-only,
# de-prioritized queue (96 nodes) -- fine for a short correctness check like
# this one, and no reason to tie up the small `coskunuzer` condo allocation
# (only 3 nodes total, 1 of them GPU) for something that doesn't need a GPU.
# This also doubles as dataset pre-caching if you skipped that step in
# setup_env.sh.

set -uo pipefail
# NOTE: can't use "$(dirname "$0")" here -- SLURM copies this script to a
# spool directory before running it, so $0 does NOT point back at the real
# slurm/ directory in the repo.
ARIA_SLURM_DIR="/groups/bcoskunuzer/avpn/ARIA/aria_benchmark/slurm"
source "${ARIA_SLURM_DIR}/common.sh"

"$PYBIN" scripts/smoke_test.py --device cpu

echo
echo "If everything above says PASS/ok: sbatch slurm/gpu_check.slurm next,"
echo "then sbatch slurm/01_baseline_search.slurm to start the real sweep."
