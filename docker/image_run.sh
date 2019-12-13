#!/bin/bash

WORK_DIR=`pwd`
docker run --name=fsl_reid --runtime=nvidia --net=host --ipc=host -it -v ${WORK_DIR}:/works/deep-person-reid -v /data/cv_hub/datasets:/works/dataset fsl/reid:v1.0 /bin/bash
