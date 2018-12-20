#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=1,2 python train.py data-bin/wmt16_en_de_bpe32k \
  -a transformer_wmt_en_de --optimizer adam --lr 0.0007 -s en -t de \
  --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy --max-update 50000 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' --clip-norm 0.0 \
  --adam-betas '(0.9, 0.98)' --save-dir checkpoints/transformer_wmt16_en_de

