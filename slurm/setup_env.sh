#!/bin/bash
# One-time environment setup. Run this ONCE, directly on the LOGIN node
# (nva220000@ganymede2.circ.utdallas.edu) -- NOT via sbatch. It needs
# internet access to pip install packages and download datasets, and
# gpu-preempt/cpu-preempt compute nodes are not guaranteed to have outbound
# internet the way the login node does.
#
# Usage:
#   cd /groups/bcoskunuzer/avpn/ARIA/aria_benchmark
#   bash slurm/setup_env.sh

set -euo pipefail

ENV_PREFIX="/groups/bcoskunuzer/avpn/conda/aria-py310"
PROJECT_DIR="/groups/bcoskunuzer/avpn/ARIA/aria_benchmark"
cd "$PROJECT_DIR"

module load cuda/12.4

echo "== Creating conda env at ${ENV_PREFIX} (skipped if it already exists) =="
if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
    conda create -y --prefix "$ENV_PREFIX" python=3.10
fi
PYBIN="${ENV_PREFIX}/bin/python"
PIPBIN="${ENV_PREFIX}/bin/pip"

echo "== Installing torch (CUDA-enabled build from PyPI, no version pin -- want"
echo "   something recent enough for H100/L40S, unlike the older torch==2.0.1"
echo "   used only for local CPU-only dev testing) + requirements.txt =="
"$PIPBIN" install --upgrade pip
"$PIPBIN" install torch
"$PIPBIN" install -r requirements.txt

echo "== Confirming torch installed correctly =="
"$PYBIN" -c "import torch, torch_geometric; print('torch', torch.__version__, '/ torch_geometric', torch_geometric.__version__)"
echo "(cuda availability isn't meaningfully testable on the login node -- no GPU"
echo " here. That check happens for real in slurm/gpu_check.slurm.)"

echo
echo "== Pre-downloading all 13 datasets (needs internet -- do this here, not"
echo "   inside a compute job) =="
"$PYBIN" -c "
from src.datasets import ALL_DATASETS, load_dataset
for d in ALL_DATASETS:
    try:
        data = load_dataset(d)
        print(f'{d}: OK, x={tuple(data.x.shape)}')
    except Exception as e:
        print(f'{d}: FAILED -- {type(e).__name__}: {e}')
"

echo
echo "== Done. Next steps: =="
echo "  sbatch slurm/smoke_test.slurm     # correctness check, dev partition, ~30s-few min"
echo "  sbatch slurm/gpu_check.slurm      # optional: confirms CUDA actually works on a GPU node"
echo "  sbatch slurm/01_baseline_search.slurm   # the real sweep -- self-chains through 02/03/04"
