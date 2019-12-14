#!/bin/sh

export PYTHONPATH=..:$PYTHONPATH
python main.py \
    --config-file ../configs/im_osnet_x1_0_softmax_256x128_amsgrad.yaml \
    --root /works/dataset \
    -t msmt17_new \
    data.height 240 \
    data.width 112 \
    test.evaluate True \
    test.dist_metric cosine \
    test.batch_size 50 \
    test.normalize_feature True
