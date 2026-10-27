#!/bin/bash
#SBATCH --job-name=dpc-python
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
set -euo pipefail
: "${TBX:?}" "${DPC:?}" "${DPC_PYTHON:?}"
export GIT_PYTHON_REFRESH=quiet WANDB_MODE=offline WANDB_SILENT=true
cd "$DPC"
echo "start $(date '+%F %T') $*"
PYTHONPATH="$DPC" "$DPC_PYTHON" "$@"
echo "end $(date '+%F %T')"
