#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=2 python generate.py data-bin/wmt14_en_de --path /data1/qspace/niucheng/elliott/fairseq/checkpoints/transformer_wmt_en_de/checkpoint_last.pt \
 --batch-size 128 --beam 5 --remove-bpe --quiet