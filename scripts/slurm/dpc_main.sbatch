#!/bin/bash
#SBATCH --job-name=dpc-main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
set -euo pipefail
: "${TBX:?}" "${DPC:?}" "${DPC_PYTHON:?}"
config=$1
mode=$2
shift 2
export GIT_PYTHON_REFRESH=quiet WANDB_MODE=offline WANDB_SILENT=true
cd "$DPC"
echo "start $(date '+%F %T') $config --mode $mode $*"
PYTHONPATH="$DPC:$TBX/scripts/slurm/dpc_site" "$DPC_PYTHON" main.py "$config" --mode "$mode" "$@"
echo "end $(date '+%F %T')"
