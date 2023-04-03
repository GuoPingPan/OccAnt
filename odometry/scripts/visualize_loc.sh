python visualize_vo_trajectory.py \
-d /home/hello/pgp_eai/ANM/data/vo_dataset \
-tro /home/yzc1/workspace/OccAnt/odometry/logs/vo_nig_wo_dropout/visualization \
--split val_mini \
--device cuda \
-epp 1000 \
-vckpt /home/yzc1/workspace/OccAnt/odometry/logs/vo_nig_wo_dropout/checkpoints/ep18_best.pth \
--embedding_size 8 \
--add_obs_noise \
--decoder_type nig \
--use_group_norm \
--use_act_embedding \
--emb_layers 2
# --emb_layers 2

# --split_action
# --use_group_norm \
# --use_act_embedding \
# --use_collision_embedding \ 

# -vckpt /home/hello/pgp_eai/easan/log/models/easan_g_nigp_ld_ac_wo_noise_f0/ep18.pth
#--en_pre
# 注释不能放在中间