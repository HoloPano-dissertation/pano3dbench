#!/bin/bash
source "$(dirname "$0")/common.sh"
: "${EXP:?set EXP to the 3D-FRONT experiment folder}"
LOGS=${LOGS:-$EXP/logs}
CONFIGS=${CONFIGS:-$EXP/slurm_configs}
mkdir -p "$LOGS" "$CONFIGS"
PANO3D_CLASSES=$("$TBX_PYTHON" -c "import json,sys;print(','.join(json.load(open(sys.argv[1]))['classes']))" "$EXP/state/classes.json")
export EXP PANO3D_CLASSES

DPC_CFG=$TBX/configs/front3d/dpc
RUNS=$TBX/configs/front3d/runs
GPU=(--gres=gpu:1 --cpus-per-task=8)

dpc_config() {
    fill "$DPC_CFG/$1" --exp "$EXP" --dpc "$DPC"
}

run_config() {
    fill "$RUNS/$1" --exp "$EXP" --dpc "$DPC"
}

case "${1:-}" in
components)
    : "${DETECTOR_INIT:?set DETECTOR_INIT to the COCO weights of Mask R-CNN}"
    detector=$(submit -J front3d-detector "${GPU[@]}" --time=1-00:00:00 "$SLURM_DIR/dpc_python.sbatch" \
        "$TBX/training/train_detector.py" "$EXP/coco" "$EXP/training/detector" --weights "$DETECTOR_INIT" ${DETECTOR_ARGS:-})
    horizonnet=$(submit -J front3d-horizonnet "${GPU[@]}" --time=1-00:00:00 "$SLURM_DIR/dpc_python.sbatch" \
        "$TBX/training/train_horizonnet.py" "$EXP/horizonnet" "$EXP/training/horizonnet" --label-format dense)
    convert=$(submit_after "$horizonnet" -J front3d-horizonnet-convert "$SLURM_DIR/dpc_python.sbatch" \
        "$TBX/training/convert_horizonnet_weights.py" "$EXP/training/horizonnet/best_valid.pth" "$EXP/training/horizonnet/for_dpc.pth")
    report "components: detector $detector, horizonnet $horizonnet, convert $convert"
    ;;
stage1)
    config=$(dpc_config training/first_stage_front3d.yaml)
    stage1=$(submit -J front3d-stage1 "${GPU[@]}" --time=1-00:00:00 "$SLURM_DIR/dpc_main.sbatch" \
        "$config" test)
    index=$(submit_after "$stage1" -J front3d-stage1-index "$SLURM_DIR/dpc_python.sbatch" \
        "$TBX/scripts/front3d/eval/index_stage1.py" "$EXP" "$EXP/dpc_stage1")
    report "stage1: inference $stage1, index $index"
    ;;
heads)
    ben_config=$(dpc_config training/bdb3d_estimation_front3d.yaml)
    ldif_config=$(dpc_config training/ldif_front3d.yaml)
    mgn_config=$(dpc_config training/mgnet_front3d.yaml)
    ben=$(submit -J front3d-ben "${GPU[@]}" "$SLURM_DIR/dpc_main.sbatch" "$ben_config" train)
    ldif=$(submit -J front3d-ldif "${GPU[@]}" "$SLURM_DIR/dpc_main.sbatch" "$ldif_config" train)
    mgn=$(submit -J front3d-mgn "${GPU[@]}" "$SLURM_DIR/dpc_main.sbatch" "$mgn_config" train)
    report "heads: ben $ben, ldif $ldif, mgn $mgn"
    ;;
gcn)
    config=$(dpc_config training/relation_scene_gcn_front3d.yaml)
    gcn=$(submit -J front3d-gcn "${GPU[@]}" "$SLURM_DIR/dpc_main.sbatch" "$config" train)
    report "gcn: $gcn"
    ;;
inference)
    names=(pano3d_front3d pano3d_front3d_s20 pano3d_front3d_im3d pano3d_front3d_total3d)
    configs=()
    for name in "${names[@]}"; do
        config=$(dpc_config "inference/$name.yaml")
        configs+=("$config")
    done
    split=$(submit -J front3d-test-split "$SLURM_DIR/dpc_python.sbatch" "$TBX/scripts/front3d/eval/make_test_split.py" "$EXP")
    jobs=()
    for index in 0 1 2 3; do
        extra=()
        if [ "$index" -lt 2 ]; then
            extra=(--model.scene_gcn.relation_adjust True)
        fi
        jobs+=("$(submit_after "$split" -J "front3d-${names[$index]}" "${GPU[@]}" --time=12:00:00 "$SLURM_DIR/dpc_main.sbatch" \
            "${configs[$index]}" qtest ${extra[@]+"${extra[@]}"})")
    done
    report "inference: split $split, dpc ${jobs[0]}, dpc_s20 ${jobs[1]}, im3d ${jobs[2]}, total3d ${jobs[3]}"
    ;;
score)
    names=(dpc dpc_s20 im3d total3d)
    roots=(pano3d_front3d pano3d_front3d_s20 pano3d_front3d_im3d pano3d_front3d_total3d)
    configs=()
    for name in "${names[@]}"; do
        config=$(run_config "${name}_rescore.yaml")
        configs+=("$config")
    done
    jobs=()
    for index in 0 1 2 3; do
        name=${names[$index]}
        convert=$(submit -J "front3d-convert-$name" "$SLURM_DIR/dpc_python.sbatch" "$TBX/scripts/dpc/dpc_batch_convert.py" \
            "$EXP/eval/manifest_test.jsonl" "$EXP/eval/pred_$name" --repo "$DPC" --pred-root "out/${roots[$index]}")
        jobs+=("$(submit_after "$convert" -J "front3d-score-$name" "$SLURM_DIR/bench_run.sbatch" "${configs[$index]}")")
    done
    report "score: dpc ${jobs[0]}, dpc_s20 ${jobs[1]}, im3d ${jobs[2]}, total3d ${jobs[3]}"
    ;;
shape)
    names=(im3d total3d)
    inference=()
    scoring=()
    for name in "${names[@]}"; do
        config=$(dpc_config "inference/shape500_$name.yaml")
        inference+=("$config")
        config=$(run_config "shape_${name}_rescore.yaml")
        scoring+=("$config")
    done
    chosen=$(submit -J front3d-shape500 "$SLURM_DIR/dpc_python.sbatch" "$TBX/scripts/front3d/eval/make_shape500.py" "$EXP")
    views=$(submit_after "$chosen" -J front3d-shape500-split \
        --wrap "cp '$EXP/eval/shape500/test_split_shape500.json' '$EXP/dpc_scenes/'")
    jobs=()
    for index in 0 1; do
        name=${names[$index]}
        infer=$(submit_after "$views" -J "front3d-shape500-$name" "${GPU[@]}" --time=12:00:00 "$SLURM_DIR/dpc_main.sbatch" \
            "${inference[$index]}" test)
        convert=$(submit_after "$infer" -J "front3d-shape-convert-$name" "$SLURM_DIR/dpc_python.sbatch" "$TBX/scripts/dpc/dpc_batch_convert.py" \
            "$EXP/eval/shape500/manifest_test.jsonl" "$EXP/eval/shape500/pred_$name" --repo "$DPC" --pred-root "out/shape500_$name" --obj-mesh-sibling obj_mesh)
        jobs+=("$(submit_after "$convert" -J "front3d-shape-score-$name" "$SLURM_DIR/bench_run.sbatch" "${scoring[$index]}")")
    done
    report "shape: views $views, im3d ${jobs[0]}, total3d ${jobs[1]}"
    ;;
*)
    echo "usage: $0 components|stage1|heads|gcn|inference|score|shape" >&2
    exit 2
    ;;
esac
