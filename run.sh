#! /bin/sh

# python submitit_finetune.py \
#     --job_dir "./finetune_checkpoints" \
#     --ngpus 1 \
#     --nodes 1 \
#     --batch_size 32 \
#     --model mae_vit_large_patch16 \
#     --finetune "./checkpoints/checkpoint-799.pth" \
#     --epochs 100 \
#     --blr 5e-4 --layer_decay 0.65 \
#     --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#     --dist_eval --data_path "./train_sets/corresponding_segment"


python submitit_pretrain.py \
    --job_dir "./checkpoints" \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 1 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path "./train_sets/one_img"