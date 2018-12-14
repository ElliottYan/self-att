#!/usr/bin/env bash
python generate.py data-bin/iwslt14.tokenized.de-en   --path checkpoints/transformer/checkpoint44.pt \
 --batch-size 128 --beam 5 --remove-bpe
