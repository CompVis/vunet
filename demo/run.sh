#! /usr/bin/env bash

python main.py  --mode "transfer" \
                --data_index ./datasets/deepfashion/index.p \
                --log_dir ./log \
                --batch_size 8 \
                --init_batches 4 \
                --checkpoint ./checkpoints/model.ckpt-100000 \
                --spatial_size 256 \
                --lr 0.001 \
                --lr_decay_begin 1000 \
                --lr_decay_end 100000 \
                --log_freq 250 \
                --ckpt_freq 1000 \
                --test_freq 1000 \
                --drop_prob 0.1 \
                --mask
                # --no-mask

