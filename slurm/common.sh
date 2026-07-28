#!/bin/bash
# Sourced (not executed) by every .slurm script in this directory. Centralizes
# environment setup so a change here (env path, CUDA module version) only
# needs to happen in one place.
#
# Mirrors the working pattern from an earlier project on this same cluster
# (GraphLabelAlignment/New_Proto/new_proto.sh): point directly at a conda
# env's python binary rather than going through `conda activate`, and
# `module load cuda/<version>` for the CUDA toolkit. Adjust ENV_PREFIX below
# if you name the env something other than aria-py310 in setup_env.sh.

set -uo pipefail

PROJECT_DIR="/groups/bcoskunuzer/avpn/ARIA/aria_benchmark"
ENV_PREFIX="/groups/bcoskunuzer/avpn/conda/aria-py310"
PYBIN="${ENV_PREFIX}/bin/python"

module purge
module load cuda/12.4 2>/dev/null || true   # harmless no-op on CPU-only partitions (dev)

mkdir -p "${PROJECT_DIR}/logs"
cd "$PROJECT_DIR"

echo "=== Job meta ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}   Job ID: ${SLURM_JOB_ID:-none}   Job name: ${SLURM_JOB_NAME:-none}"
echo "Using python: ${PYBIN}"
"$PYBIN" -V
echo
