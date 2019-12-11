# FROM
FROM nvidia/cuda:10.1-cudnn7-devel-centos7

# dependencies
RUN yum install -y epel-release \
    && yum install -y wget which gcc-c++ make cmake3 git \
    && yum install -y libXext libSM libXrender boost-devel \
    && ln -s /usr/bin/cmake3 /usr/bin/cmake

# miniconda
WORKDIR /opt
ADD http://pub-shyc2.s3.360.cn/vision/miniconda3-4.7.10.sh .
RUN /bin/bash miniconda3-4.7.10.sh -bs -p /opt/conda \
    && export PATH=/opt/conda/bin:$PATH \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
    && conda install conda \
    && conda install torchvision \
    && rm -rf miniconda3-4.7.10.sh
ENV PATH=/usr/local/bin:/opt/conda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/lib64:/usr/local/lib:/usr/local/lib64:/opt/conda/lib:$LD_LIBRARY_PATH \
    CPLUS_INCLUDE_PATH=/usr/local/include:/opt/conda/include:$CPLUS_INCLUDE_PATH

COPY ./requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.douban.com/simple \
    && rm requirements.txt
