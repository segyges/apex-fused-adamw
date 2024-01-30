# Copied and heavily modified on 1/
#
# Copyright (c) 2024, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# metainformation
LABEL org.opencontainers.image.version = "2.0"
LABEL org.opencontainers.image.authors = "segygs@gmail.com"
LABEL org.opencontainers.image.source = "https://www.github.com/segyges/not-nvidia-apex"
LABEL org.opencontainers.image.licenses = "BSD 3-Clause"
LABEL org.opencontainers.image.base.name="docker.io/nvidia/cuda:12.1.1-devel-ubuntu22.04"


#### System package (uses default Python 3 version in Ubuntu 22.04)
# I have no idea which of these are necessary and have opted to mostly leave them alone.
# Probably, most of them are not necessary.
RUN apt-get update -y && \
	apt-get install -y \
	git python3-dev libpython3-dev python3-pip sudo pdsh \
	tmux zstd software-properties-common build-essential autotools-dev \
	nfs-common pdsh cmake g++ gcc curl wget vim less unzip htop iftop iotop ca-certificates ssh \
	rsync iputils-ping net-tools libcupti-dev libmlx4-1 infiniband-diags ibutils ibverbs-utils \
	rdmacm-utils perftest rdma-core nano && \
	update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
	update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
	python -m pip install --upgrade pip && \
	python -m pip install packaging ninja

#### Python packages
RUN python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY . /mnt/build/
WORKDIR /mnt/build/
## Install APEX
RUN python -m pip wheel -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings --global-option=--cpp_ext --config-settings --global-option=--cuda_ext ./

