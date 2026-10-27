#!/bin/bash
#SBATCH --job-name=bench-run
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
set -euo pipefail
: "${TBX:?}"
cd "$TBX"
PYTHONPATH=src "${TBX_PYTHON:-$TBX/.venv/bin/python}" -m pano3dbench run --config "$1"
