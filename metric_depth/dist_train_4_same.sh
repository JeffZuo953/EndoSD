#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")

epoch=120
bs=24
gpus=1
lr=0.000005
encoder=vits
dataset=combined
min_depth=1e-6
pretrained_from=/data/depthanything/depth_anything_v2_${encoder}.pth
save_path=/data/train_4_dataset/same_maxdepth_full_${now}
img_size=475
max_depth=0.2
num_workers=32

mkdir -p $save_path

python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    ./train_4_same.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset --num-workers $num_workers \
    --img-size $img_size --min-depth $min_depth --max-depth $max_depth --pretrained-from $pretrained_from --no-compile \
    --port 20596 2>&1 | tee -a $save_path/$now.log
