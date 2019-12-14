# FROM
FROM nvidia/cuda:10.1-cudnn7-devel-centos7

# dependencies
RUN yum install -y epel-release \
    && yum install -y wget which gcc-c++ make cmake3 git \
    && yum install -y autoconf automake libtool zlib-devel \
    && yum install -y leveldb-devel snappy-devel hdf5-devel lmdb-devel \
    && yum install -y libXext libSM libXrender \
    && ln -s /usr/bin/cmake3 /usr/bin/cmake

# miniconda
WORKDIR /opt
ADD http://pub-shyc2.s3.360.cn/vision/miniconda3-4.7.10.sh .
RUN /bin/bash miniconda3-4.7.10.sh -bs -p /opt/conda \
    && export PATH=/opt/conda/bin:$PATH \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
    && conda install conda \
    && conda install numpy boost opencv openblas cudnn nccl libgcc scikit-image \
    && pip install torchvision -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com/pypi/simple/ \
    && rm -rf miniconda3-4.7.10.sh
ENV PATH=/usr/local/bin:/opt/conda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/lib64:/usr/local/lib:/usr/local/lib64:/opt/conda/lib:$LD_LIBRARY_PATH \
    CPLUS_INCLUDE_PATH=/usr/local/include:/opt/conda/include:$CPLUS_INCLUDE_PATH

RUN wget https://github.com/protocolbuffers/protobuf/archive/v3.10.1.tar.gz \
    && tar xvf v3.10.1.tar.gz \
    && pushd protobuf-3.10.1 \
    && ./autogen.sh && ./configure && make -j && make install \
    && popd \
    && pushd protobuf-3.10.1/python \
    && python setup.py build \
    && mv build/lib ../protobuf-python-lib \
    && popd \
    && wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/google-glog/glog-0.3.3.tar.gz \
    && tar xvf glog-0.3.3.tar.gz \
    && pushd glog-0.3.3 \
    && ./configure && make -j && make install \
    && popd \
    && wget https://github.com/schuhschuh/gflags/archive/v2.2.2.tar.gz \
    && tar xvf v2.2.2.tar.gz \
    && mkdir gflags-2.2.2/build \
    && pushd gflags-2.2.2/build && cmake -DBUILD_SHARED_LIBS=on .. && make -j && make install \
    && popd \
    && wget https://github.com/glennrp/libpng/archive/v1.6.35.tar.gz \
    && tar xvf v1.6.35.tar.gz \
    && pushd libpng-1.6.35 \
    && ./configure && make -j && make install \
    && popd \
    && rm -rf v3.10.1.tar.gz protobuf-3.10.1 glog-0.3.3.tar.gz glog-0.3.3 v2.2.2.tar.gz gflags-2.2.2 v1.6.35.tar.gz libpng-1.6.35

#COPY ./Makefile.config .
# ADD http://pub-shyc2.s3.360.cn/vision/caffe-master.tar.gz .
# RUN mkdir caffe && tar xvf caffe-master.tar.gz -C caffe --strip-components 1 \
    # && rm -rf caffe-master.tar.gz \
    # && pushd caffe \
    # && mv ../Makefile.config . \
    # && make -j && make pycaffe \
    # && popd
ENV PYTHONPATH=/opt/protobuf-python-lib:/opt/caffe/python:$PYTHONPATH
