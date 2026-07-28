#!/bin/bash
#SBATCH -J aria-gpu-check
#SBATCH -p coskunuzer #coskunuzer # swap to -p coskunuzer if the lab's condo GPU nodes are free (check: sinfo -p coskunuzer) -- not subject to preemption
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH -o /groups/bcoskunuzer/avpn/ARIA/aria_benchmark/logs/%x-%j.out
#SBATCH --mail-user=neelam.akula@utdallas.edu
#SBATCH --mail-type=END,FAIL

# Optional but recommended before the real sweep: confirms torch actually
# sees a GPU on this cluster's node type (H100/L40S) and that a training
# pass runs on it, not just on CPU (smoke_test.slurm alone doesn't prove this).

set -uo pipefail
# NOTE: can't use "$(dirname "$0")" here -- SLURM copies this script to a
# spool directory before running it, so $0 does NOT point back at the real
# slurm/ directory in the repo.
ARIA_SLURM_DIR="/groups/bcoskunuzer/avpn/ARIA/aria_benchmark/slurm"
source "${ARIA_SLURM_DIR}/common.sh"

nvidia-smi
"$PYBIN" -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

"$PYBIN" scripts/smoke_test.py --device cuda \
    --datasets cornell wisconsin roman-empire \
    --models mlp gcn gps aria aria_moe aria_ensemble
