#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=1,2 python train.py data-bin/wmt14_en_de \
  -a general_transformer_multidim_wmt_en_de --optimizer adam --lr 0.0005 -s de -t en \
  --label-smoothing 0.1 --dropout 0.3 --max-tokens 1000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --max-update 50000 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --adam-betas '(0.9, 0.98)' --save-dir checkpoints/general_transformer_multi_wmt_en_de
