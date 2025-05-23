#!/bin/bash

python -u train.py \
  --init_method "tcp://127.0.0.1:$PORT" \
  -c ./configs/transformer.yaml \
  --outdir "./output" \
  --world_size 1 \
  --desc "v2-newposition2-4wdataset-test-valid-2023" \
  --resume False \
  --smear 0 \
  --dos_minmax False \
  --dos_zscore False \
  --scale_factor 1.0 \
  --apply_log False \
  --seed 42