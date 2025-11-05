#!/bin/bash

python evaluate_lora.py \
    --encoder vits \
    --img-size 518 \
    --save-path /root/Depth-Anything-V2/exp/vits_fine_tuned \
    --load-from /root/Depth-Anything-V2/exp/c3vd_lora_20250509_003418/best_lora_abs_rel.pth \
    --max-depth 0.1 \
    --min-depth 1e-6\
    --dataset c3vd \
    --lora_r 4 \
    --lora_alpha 4 \
    --lora_dropout 0.0 \
    --lora_bias none
