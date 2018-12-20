#!/usr/bin/env bash
cd ..
task=train_general_transformer_iwslt14
CUDA_VISIBLE_DEVICES=2 python train.py data-bin/iwslt14.tokenized.de-en \
  -a general_transformer_iwslt_de_en --optimizer adam --lr 0.0005 -s de -t en \
  --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --max-update 50000 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --save-interval 5 \
  --adam-betas '(0.9, 0.98)' --save-dir checkpoints/general_transformer_iwslt14_drop_head \
  --attention_type 'MultiheadAttentionDropoutHead'
