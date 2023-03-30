LOG_DIR=${1}

export CUDA_VISIBLE_DEVICES=3

python3 train_vo.py \
-d /home/yzc1/workspace/OccAnt/data/vo_dataset \
--split train  \
--epochs 20 \
--batch_size 128 \
--device cuda \
--vonet_log_interval 200 \
--save_model_epochs 2 \
--save_model \
--lr 5e-4 \
--log_dir logs/vo_base_act_colli_emb \
--embedding_size 8 \
--use_dropout \
--add_obs_noise \
--decoder_type base \
--emb_layers 2 \
--use_group_norm \
--use_act_embedding \
--use_collision_embedding \

# --split_action
 

# -vckpt /home/hello/pgp_eai/easan/log/models/easan_g_nigp_ld_ac_wo_noise_f0/ep18.pth
#--en_pre
# 注释不能放在中间