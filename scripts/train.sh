#!/bin/sh

export PYTHONPATH=..:$PYTHONPATH
python main.py \
  --config-file ../configs/im_osnet_x1_0_softmax_256x128_amsgrad.yaml \
  -s market1501 \
  -t market1501 \
  --transforms random_flip color_jitter \
  --root /works/dataset
