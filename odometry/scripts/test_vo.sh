OUT_DIR=${1}

python test/test_vo.py \
-d /home/yzc1/workspace/OccAnt/data/vo_dataset \
--split val_mini \
--device cuda \
-vckpt /home/hello/pgp_eai/easan/log/models/easan_g_nigp_ld_ac_wo_noise_f20/ep18.pth \
-tdo ${OUT_DIR} \
--use_group_norm \
-ntp 10

