#!/bin/bash

WORK_DIR=$(pwd)
docker run \
    --name=fsl_reid_qhaicv \
    --runtime=nvidia \
    --net=host \
    --ipc=host \
    -it \
    -v ${WORK_DIR}:/works/deep-person-reid \
    -v /home/fushilian/caffe-docker:/opt/caffe \
    -v /data/cv_hub/datasets:/works/dataset \
    fsl/caffe-gpu:v1.0 /bin/bash
