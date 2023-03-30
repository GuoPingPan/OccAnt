LOG_DIR=${1}

# export CUDA_VISIBLE_DEVICES=0,1


torchrun --nproc_per_node=4 train_vo.py \
-d /home/yzc1/workspace/OccAnt/data/vo_dataset \
--split val_mini  \
--epochs 20 \
--batch_size 32 \
--device cuda \
--vonet_log_interval 200 \
--save_model_epochs 2 \
--save_model \
--lr 5e-4 \
--log_dir ${LOG_DIR} \
--embedding_size 8 \
--use_dropout \
--add_obs_noise \
--decoder_type base
# --emb_layers 2

# --split_action
# --use_group_norm \
# --use_act_embedding \
# --use_collision_embedding \ 

# -vckpt /home/hello/pgp_eai/easan/log/models/easan_g_nigp_ld_ac_wo_noise_f0/ep18.pth
#--en_pre
# 注释不能放在中间