# Base image
FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

# setup environment
ENV TERM xterm
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python3.8/dist-packages/torch/lib/
ENV PYTHONPATH=/depoco/submodules/ChamferDistancePytorch/

# Provide a data directory to share data across docker and the host system
RUN mkdir -p /data

# mirrors.aliyun.com / mirrors.tuna.tsinghua.edu.cn

# RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list 
RUN rm -rf /var/lib/apt/lists/* && apt-get clean
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list 
# RUN sed -i -re 's/de.archive.ubuntu.com/archive.ubuntu.com/g' /etc/apt/sources.list
# RUN apt-get update

RUN apt-get clean && apt-get update && apt-get install -y software-properties-common 
# RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update 

# RUN apt-get install -y software-properties-common 
    # && add-apt-repository ppa:deadsnakes/ppa && apt-get update 

# Install system packages
RUN rm -rf /var/lib/apt/lists/* && apt-get clean && apt-get update 
RUN apt-get clean && apt-get update && apt-get install --no-install-recommends -y \
    build-essential 

RUN apt-get install --no-install-recommends -y \
    cmake 
RUN apt-get install --no-install-recommends -y \
    git     
RUN apt-get install --no-install-recommends -y \
    libeigen3-dev 
RUN apt-get install --no-install-recommends -y \
    libgl1-mesa-glx 
RUN apt-get install --no-install-recommends -y \
    libusb-1.0-0-dev 
RUN apt-get install --no-install-recommends -y \
    ninja-build 
RUN apt-get install --no-install-recommends -y \
    pybind11-dev 
RUN apt-get install --no-install-recommends -y \
    python3 
RUN apt-get install --no-install-recommends -y \
    python3-dev 
RUN apt-get install --no-install-recommends -y \
    python3-pip 
RUN apt-get install --no-install-recommends -y \
    vim
RUN rm -rf /var/lib/apt/lists/*

# Install Pytorch with CUDA 11 support
# RUN pip3 install \
#     torch==1.7.1 \
#     -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip3 install \    
#     torchvision==0.8.2+cu110 \
#     torchaudio==0.7.2 \
#     -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install \
    numpy  -i https://pypi.tuna.tsinghua.edu.cn/simple


# Install Pytorch with CUDA 11 support
RUN pip3 install torch===1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html -i https://mirrors.aliyun.com/pypi/simple/ some-package

RUN pip3 install torchvision===0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html -i https://mirrors.aliyun.com/pypi/simple/ some-package

# RUN pip3 install \
#     torch==1.7.1 
#     # -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip3 install \    
#     torchvision==0.8.2+cu110 \
#     torchaudio==0.7.2 
#     # -f https://download.pytorch.org/whl/torch_stable.html

# Install python dependencies
RUN pip3 install \
    open3d  \
    tensorboard \
    ruamel.yaml \
    jupyterlab \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple



# Copy the libary to the docker image
COPY ./ depoco/

# Install depoco and 3rdparty dependencies
RUN cd depoco/ && pip3 install -U -e .

RUN cd depoco/submodules/octree_handler && pip3 install -U .

ENV TORCH_CUDA_ARCH_LIST="7.5"
RUN cd depoco/submodules/ChamferDistancePytorch/chamfer3D/ && pip3 install -U . 2>/dev/null



WORKDIR /depoco/depoco
