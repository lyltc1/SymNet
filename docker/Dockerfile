ARG PYTORCH="1.13.1"
ARG CUDA="11.6"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel


# mirrors
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse" >>/etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse" >>/etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse" >>/etc/apt/sources.list && \
    pip config set global.index-url https://mirrors.zju.edu.cn/pypi/web/simple

# git
RUN apt-get update && \
    apt-get install -y git build-essential unzip

# dependence
RUN apt-get install -y build-essential unzip

RUN pip install scipy kiwisolver==1.3.1 matplotlib imageio pypng Cython==0.29.24 PyOpenGL==3.1.0 triangle>=20190115.2 glumpy Pillow>=8.2.0 vispy>=0.6.5 openmim && \
    mim install "mmcv>=2.0.0rc1" && \
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' && \
    pip install pytorch-lightning==1.8.4.post0 imgaug moderngl trimesh transforms3d numba opencv-python-headless

# clone the code
WORKDIR /home
RUN git clone https://github.com/lyltc1/SymNet.git && \
    cd SymNet && \
    git checkout -b mmcv2 origin/mmcv2
WORKDIR /home/SymNet/bop_toolkit
RUN pip install -e .
    
# mirrors
RUN pip config unset global.index-url
RUN mv /etc/apt/sources.list.bak /etc/apt/sources.list

# Clean everything
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
WORKDIR /home/SymNet

