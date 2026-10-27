set -euo pipefail
TBX=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SLURM_DIR=$TBX/scripts/slurm
: "${DPC:?set DPC to the DeepPanoContext clone}"
: "${DPC_PYTHON:?set DPC_PYTHON to the python of the Pano3D environment}"
TBX_PYTHON=${TBX_PYTHON:-$TBX/.venv/bin/python}
export TBX DPC DPC_PYTHON TBX_PYTHON

submit_after() {
    local after=$1
    shift
    local dependency=()
    if [ -n "$after" ]; then
        dependency=(--dependency="afterok:$after")
    fi
    sbatch --parsable --output="$LOGS/%x-%j.out" ${dependency[@]+"${dependency[@]}"} "$@"
}

submit() {
    submit_after "${AFTER:-}" "$@"
}

fill() {
    local target=$CONFIGS/$(basename "$1")
    "$TBX_PYTHON" "$SLURM_DIR/fill_config.py" "$1" "$target" "${@:2}" >/dev/null
    echo "$target"
}

report() {
    echo "$(date '+%F %T') $*" | tee -a "$LOGS/submitted.txt"
}
