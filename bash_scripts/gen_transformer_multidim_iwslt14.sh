#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=1 python generate.py data-bin/iwslt14.tokenized.de-en   --path checkpoints/transformer_multi/checkpoint_last.pt \
 --batch-size 32 --beam 5 --remove-bpe
