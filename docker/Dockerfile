# https://www.learnopencv.com/install-opencv3-on-ubuntu
# https://docs.opencv.org/3.4/d6/d15/tutorial_building_tegra_cuda.html

ARG CUDA_VERSION=10.1
ARG CUDNN_VERSION=7

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu18.04

ARG PYTHON_VERSION=3.8


# Install all dependencies 
    apt-get -y update --fix-missing && \
    apt-get -y install --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
		python-opencv \
    && \


# install python dependencies
    sysctl -w net.ipv4.ip_forward=1 && \
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \
    pip${PYTHON_VERSION} install numpy && \

# install CMake
    cd /home && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.7/cmake-3.14.7-Linux-x86_64.tar.gz && \
    tar -xvf cmake-3.14.7-Linux-x86_64.tar.gz && \
    mv cmake-3.14.7-Linux-x86_64 /opt && \
    echo "export PATH=/opt/cmake-3.14.7-Linux-x86_64/bin:$PATH" >> /etc/profile && \
    . /etc/profile && \

# # install nccl
#     tar -xvf nccl.tar.gz && \
#     cd /home/nccl && make -j install && cd .. && rm -rf nccl && \
#     rm nccl.tar.gz && \
    rm cmake-3.14.7-Linux-x86_64.tar.gz && \
    cd ..  && \


# Set the default python and install PIP packages
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \

# Call default command.
    python --version && \
    #python -c "import cv2 ; print(cv2.__version__)"




