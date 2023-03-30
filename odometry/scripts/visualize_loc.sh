

OUT_DIR=${1}

python test/visualize_vo_trajectory.py \
-d /home/hello/pgp_eai/ANM/data/vo_dataset \
-tro ${OUT_DIR} \
--split val_mini \
--device cuda \
-epp 1000 \
-vckpt /home/hello/pgp_eai/easan/log/models/easan_g_nigp_ld_ac_wo_noise_f20/ep17_best.pth \
--use_group_norm
