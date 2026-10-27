#!/bin/bash
source "$(dirname "$0")/common.sh"
LOGS=${LOGS:-$DPC/logs}
CONFIGS=${CONFIGS:-$DPC/slurm_configs}
BENCH=$DPC/data/igibson_bench
mkdir -p "$LOGS" "$CONFIGS"

DPC_CFG=$TBX/configs/igibson/dpc
RUNS=$TBX/configs/igibson/runs
GPU=(--gres=gpu:1 --cpus-per-task=8)
SPLIT=(--data.split data/igibson)

dpc_config() {
    cp "$DPC_CFG/$1" "$DPC/configs/$(basename "$1")"
    echo "configs/$(basename "$1")"
}

run_config() {
    fill "$RUNS/$1" --dpc "$DPC"
}

case "${1:-}" in
heads)
    targets=$(submit -J igibson-mgn-targets --cpus-per-task=8 "$SLURM_DIR/dpc_python.sbatch" \
        "$TBX/scripts/igibson/mgn_make_gt.py" "$DPC" --processes 8)
    mgn=$(submit_after "$targets" -J igibson-mgn "${GPU[@]}" "$SLURM_DIR/dpc_main.sbatch" "$(dpc_config training/mgnet_igibson.yaml)" qtrain)
    ldif=$(submit -J igibson-ldif "${GPU[@]}" "$SLURM_DIR/dpc_main.sbatch" "$(dpc_config training/ldif_igibson_scratch.yaml)" qtrain)
    report "heads: mgn targets $targets, mgn $mgn, ldif $ldif"
    ;;
inference)
    im3d_config=$(dpc_config inference/pano3d_igibson_im3d.yaml)
    total3d_config=$(dpc_config inference/pano3d_igibson_total3d.yaml)
    dpc=$(submit -J igibson-dpc "${GPU[@]}" --time=12:00:00 "$SLURM_DIR/dpc_main.sbatch" \
        configs/pano3d_igibson.yaml qtest "${SPLIT[@]}" --log.path out/pano3d)
    s20=$(submit -J igibson-dpc-s20 "${GPU[@]}" --time=12:00:00 "$SLURM_DIR/dpc_main.sbatch" \
        configs/pano3d_igibson.yaml qtest "${SPLIT[@]}" --model.scene_gcn.optimize_steps 20 --log.path out/pano3d_dpc_s20)
    im3d=$(submit -J igibson-im3d "${GPU[@]}" --time=12:00:00 "$SLURM_DIR/dpc_main.sbatch" \
        configs/pano3d_igibson.yaml qtest "${SPLIT[@]}" --model.scene_gcn.relation_adjust False --log.path out/pano3d_im3d)
    trained=$(submit -J igibson-im3d-trained "${GPU[@]}" --time=12:00:00 "$SLURM_DIR/dpc_main.sbatch" \
        "$im3d_config" qtest "${SPLIT[@]}" --model.scene_gcn.relation_adjust False --log.path out/pano3d_im3d_trained)
    total3d=$(submit -J igibson-total3d "${GPU[@]}" --time=12:00:00 "$SLURM_DIR/dpc_main.sbatch" \
        "$total3d_config" qtest "${SPLIT[@]}" --log.path out/pano3d_total3d)
    report "inference: dpc $dpc, dpc_s20 $s20, im3d $im3d, im3d_trained $trained, total3d $total3d"
    ;;
score)
    truth=$(submit -J igibson-ground-truth "$SLURM_DIR/dpc_python.sbatch" \
        "$TBX/scripts/igibson/dpc_build_igibson_manifest.py" "$DPC" "$BENCH" "$DPC/data/igibson/test.json")
    jobs=()
    for pair in pred:pano3d:dpc pred_dpc_s20:pano3d_dpc_s20:dpc_s20 pred_im3d:pano3d_im3d:im3d \
                pred_im3d_trained:pano3d_im3d_trained:im3d_trained pred_total3d:pano3d_total3d:total3d; do
        IFS=: read -r out root name <<< "$pair"
        convert=$(submit_after "$truth" -J "igibson-convert-$name" "$SLURM_DIR/dpc_python.sbatch" \
            "$TBX/scripts/dpc/dpc_batch_convert.py" "$BENCH/manifest.jsonl" "$BENCH/$out" --repo "$DPC" --pred-root "out/$root")
        jobs+=("$(submit_after "$convert" -J "igibson-score-$name" "$SLURM_DIR/bench_run.sbatch" "$(run_config "${name}_rescore.yaml")")")
    done
    report "score: ground truth $truth, dpc ${jobs[0]}, dpc_s20 ${jobs[1]}, im3d ${jobs[2]}, im3d_trained ${jobs[3]}, total3d ${jobs[4]}"
    ;;
*)
    echo "usage: $0 heads|inference|score" >&2
    exit 2
    ;;
esac
