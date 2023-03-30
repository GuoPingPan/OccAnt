# python generate_dataset.py \
# -d None \
# -agc 0 \
# --num_processes_per_gpu 3 \
# --split val_mini \
# -pps 2000 \
# -epp 500 \

# python generate_dataset.py \
# -d None \
# -agc 0 \
# --num_processes_per_gpu 14 \
# --split val \
# -pps 2000 \
# -epp 500 \


python generate_dataset.py \
-d None \
-agc 0 \
--num_processes_per_gpu 18 \
--split train \
-pps 2000 \
-epp 500 \


