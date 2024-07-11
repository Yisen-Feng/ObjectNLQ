
#### training
config_file=$1
exp_id=$2
device_id=$3

CUDA_VISIBLE_DEVICES=${device_id} python train_single.py ${config_file} \
--output ${exp_id} --mode=train \
${@:4}