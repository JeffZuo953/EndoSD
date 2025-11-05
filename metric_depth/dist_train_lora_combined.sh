#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=120
bs=45 # Updated batch size
gpus=1
lr=0.000005
encoder=vits
dataset=combined # Updated dataset
img_size=518
min_depth=0.000001 # Updated min depth
max_depth=0.2 # Updated max depth
pretrained_from=/root/dpt/checkpoints/depth_anything_v2_metric_hypersim_vits.pth # Updated pretrained model path
save_path=/data/train_combined_lora_${now} # Updated save path, added _lora_ suffix and timestamp

# Re-added LoRA parameters
lora_r=16        # LoRA rank (set to 0 to disable LoRA)
lora_alpha=16    # LoRA alpha (typically same as lora_r or 1)
lora_dropout=0.0 # LoRA dropout
lora_bias='none' # LoRA bias ('none', 'lora_only', 'all')
lora_head_lr_multiplier=10.0 # DPT head learning rate multiplier

mkdir -p $save_path # Create save path

# Reverted execution command to use torch.distributed.launch and train_lora.py, re-added LoRA args
python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    ./train_lora.py \
    --epochs $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --lora_bias $lora_bias \
    --lora_head_lr_multiplier $lora_head_lr_multiplier \
    --img-size $img_size --min-depth $min_depth --max-depth $max_depth --pretrained-from $pretrained_from \
    --port 20596 2>&1 | tee -a $save_path/$now.log
