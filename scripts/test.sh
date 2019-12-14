export PYTHONPATH=..:$PYTHONPATH
python main.py \
  --root /works/dataset \
  data.height 240 \
  data.width 112 \
  test.evaluate True \
  test.dist_metric cosine \
  test.batch_size 50 \
  test.normalize_feature True \
  caffe.prototxt BODY_FEATURE.prototxt \
  caffe.weights BODY_FEATURE.caffemodel \
  use_gpu False
