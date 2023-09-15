FROM iaslab/env:cuda11.6-torch1.12-0.18.5

COPY dockerfile_scripts /tmp/dockerfile_scripts

# ---- install ninja
RUN apt-get update && apt-get install -y apt-utils
RUN apt-get install -y ca-certificates wget sudo ninja-build

# RUN pip config set global.index-url https://mirrors.zju.edu.cn/pypi/web/simple

# ---- install tensorboard cmake
RUN pip install cmake cython

# ---- install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
# ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ARG TORCH_CUDA_ARCH_LIST=7.5;8.6
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
RUN pip install -e detectron2
ENV FVCORE_CACHE="/tmp"

# ---- dependences
RUN chmod a+x /tmp/dockerfile_scripts/gdrnet/install_deps.sh &&\
    /tmp/dockerfile_scripts/gdrnet/install_deps.sh

RUN chmod a+x /tmp/dockerfile_scripts/bop_toolkit/install_deps.sh &&\
    /tmp/dockerfile_scripts/bop_toolkit/install_deps.sh

RUN pip install 'git+https://github.com/thodan/bop_toolkit.git'


# 删除临时文件
RUN rm -r /tmp/*
