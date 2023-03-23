SPLIT=${1}

python occant_baselines/generate_topdown_maps/generate_occant_gt_maps.py \
    --config-path occant_baselines/generate_topdown_maps/configs/occant_gt/gibson_${SPLIT}.yaml \
    --save-dir data/datasets/exploration/gibson/v1/${SPLIT}/occant_gt_maps \
    --global-map-size 961