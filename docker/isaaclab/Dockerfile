# syntax=docker/dockerfile:1
#
# Isaac Sim 5.1.0 + Isaac Lab 2.3.1 + ROS 2 Humble
# Ubuntu 22.04 / CUDA 12.8 / Python 3.11 / PyTorch 2.7.0 (cu128)
# 가상환경: miniconda `isaac_lab`
#
# 빌드:  DOCKER_BUILDKIT=1 docker build -t rospider/isaaclab:2.3.1 docker/isaaclab
#
# isaacsim[all,extscache] 는 내려받는 양이 10GB 를 넘습니다. 빌드에 30~90분 걸립니다.

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE} AS base

# ---------------------------------------------------------------- 버전 핀
ARG PYTHON_VERSION=3.11
ARG CONDA_ENV=isaac_lab
ARG ISAACSIM_VERSION=5.1.0
ARG ISAACLAB_REF=v2.3.1
ARG TORCH_VERSION=2.7.0
ARG TORCHVISION_VERSION=0.22.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG ROS_DISTRO=humble
ARG TZ=Asia/Seoul

# ROS 2 패키지 목록. 가볍게 가려면 --build-arg ROS_APT_PACKAGES="ros-humble-ros-base"
ARG ROS_APT_PACKAGES="ros-humble-ros-base ros-humble-rmw-fastrtps-cpp ros-humble-rviz2 ros-humble-vision-msgs ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox python3-colcon-common-extensions"

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=${TZ} \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    CONDA_DIR=/opt/conda \
    ISAAC_CONDA_ENV=${CONDA_ENV} \
    ISAACLAB_PATH=/opt/IsaacLab \
    ROS_DISTRO=${ROS_DISTRO} \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    ROS_DOMAIN_ID=0 \
    OMNI_KIT_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ------------------------------------------------- 시스템 패키지 + 로케일
# build-essential / cmake / git / libglib2.0-0 / ncurses-term: Isaac Lab 공식 Dockerfile.base 목록
# 나머지는 Kit(Omniverse) 런타임이 요구하는 GL·Vulkan·X11 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git git-lfs \
      curl wget gnupg2 ca-certificates \
      locales tzdata ncurses-term less nano unzip \
      libglib2.0-0 libgl1 libglu1-mesa libegl1 libgles2 \
      libvulkan1 vulkan-tools libxkbcommon-x11-0 \
      libx11-6 libxext6 libxrender1 libxrandr2 libxinerama1 libxcursor1 libxi6 \
      libxtst6 libsm6 libice6 libxt6 libfontconfig1 libfreetype6 \
      libasound2 libatomic1 libusb-1.0-0 \
    && ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone \
    && locale-gen en_US en_US.UTF-8 && update-locale LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------ ROS 2 Humble
# Ubuntu 22.04(jammy) 의 시스템 파이썬은 3.10 입니다. 즉 /opt/ros/humble 의 rclpy 는
# conda 의 python 3.11 에서 import 되지 않습니다. 설정 실수가 아니라 구조적 제약입니다.
# Isaac Sim 은 번들된 내부 Humble 라이브러리로, ROS 노드는 시스템 3.10 으로 각각 돌리고
# DDS 로 붙입니다. isaac-shell / ros-shell 이 두 쉘을 분리해 줍니다.
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
      > /etc/apt/sources.list.d/ros2.list \
 && apt-get update && apt-get install -y --no-install-recommends ${ROS_APT_PACKAGES} \
 && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------- 캐시 디렉터리
RUN mkdir -p /root/.cache/ov /root/.cache/pip /root/.cache/nvidia/GLCache \
             /root/.nv/ComputeCache /root/.nvidia-omniverse/logs \
             /root/.local/share/ov/data /root/Documents

# ------------------------------------------------------------------- 쉘 진입점
COPY scripts/ /opt/isaaclab-scripts/
COPY steps/   /opt/isaaclab-steps/
RUN chmod +x /opt/isaaclab-scripts/entrypoint.sh \
             /opt/isaaclab-scripts/isaac-shell \
             /opt/isaaclab-scripts/ros-shell \
 && ln -sf /opt/isaaclab-scripts/isaac-shell /usr/local/bin/isaac-shell \
 && ln -sf /opt/isaaclab-scripts/ros-shell   /usr/local/bin/ros-shell \
 && chmod +x /opt/isaaclab-steps/*.sh \
 && echo 'source /opt/isaaclab-scripts/isaac-env.sh' >> /root/.bashrc

WORKDIR /workspace

# Isaac Sim WebRTC 스트리밍(헤드리스) 포트
EXPOSE 8211/udp 49100/tcp

ENTRYPOINT ["/opt/isaaclab-scripts/entrypoint.sh"]
CMD ["bash"]

# ============================================================================
#  base 스테이지 끝.
#    docker build --target base   -> 시스템 패키지 + ROS 2 만 있는 가벼운 이미지.
#                                    steps/ 를 한 단계씩 돌려 설치할 때 씁니다.
#    docker build                 -> 아래 full 스테이지까지. 전부 구워진 이미지.
# ============================================================================

FROM base AS full

ARG CONDA_ENV=isaac_lab
ARG PYTHON_VERSION=3.11
ARG ISAACSIM_VERSION=5.1.0
ARG ISAACLAB_REF=v2.3.1
ARG TORCH_VERSION=2.7.0
ARG TORCHVISION_VERSION=0.22.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

# -------------------------------------------------------------- Miniconda
# conda 24.x+ 는 Anaconda 기본 채널(repo.anaconda.com/pkgs/*)에 ToS 동의를 요구합니다.
# 동의 없이 무인 실행하면 CondaToSNonInteractiveError 로 빌드가 멈춥니다. 게다가 기본 채널은
# 규모 있는 조직에 유상 라이선스를 요구하므로, conda-forge 단독으로 갑니다.
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash /tmp/miniconda.sh -b -u -p ${CONDA_DIR} \
 && rm /tmp/miniconda.sh \
 && ${CONDA_DIR}/bin/conda config --set always_yes true \
 && ${CONDA_DIR}/bin/conda config --set auto_activate_base false \
 && ${CONDA_DIR}/bin/conda config --add channels conda-forge \
 && (${CONDA_DIR}/bin/conda config --remove channels defaults || true) \
 && ${CONDA_DIR}/bin/conda create -n ${CONDA_ENV} python=${PYTHON_VERSION} \
      -c conda-forge --override-channels \
 && ${CONDA_DIR}/bin/conda clean -afy \
 && ${CONDA_DIR}/bin/conda init bash

# 아래 RUN 들은 전부 `conda activate isaac_lab` 안에서 실행됩니다.
RUN --mount=type=cache,target=/root/.cache/pip \
    source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} \
 && python -m pip install --upgrade pip --root-user-action=ignore \
 && python -c "import sys; assert sys.version_info[:2]==(3,11), sys.version; print('python', sys.version.split()[0])"

# ------------------------------------------------------------- Isaac Sim 5.1
# pip 설치는 GLIBC 2.35+ 를 요구합니다. Ubuntu 22.04 가 정확히 2.35 라 하한선에 걸칩니다.
RUN --mount=type=cache,target=/root/.cache/pip \
    ldd --version | head -n1 \
 && source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} \
 && pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" --extra-index-url https://pypi.nvidia.com

# --------------------------------------------------------- PyTorch 2.7.0 cu128
# isaacsim 이 끌고 온 torch 를 지정 버전으로 덮어씁니다.
RUN --mount=type=cache,target=/root/.cache/pip \
    source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} \
 && pip install -U "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url ${TORCH_INDEX}

# ------------------------------------------------------------ Isaac Lab 2.3.1
# flatdict==4.0.1 은 sdist 뿐이라 소스 빌드가 필요한데, 그 setup.py 가 import 하는
# pkg_resources 는 setuptools 82.0.0 에서 제거됐습니다. pip 의 빌드 격리 환경은 항상
# 최신 setuptools 를 받아오므로 그대로 두면 isaaclab 설치가 여기서 깨집니다
# (PIP_CONSTRAINT 는 격리 overlay 에 적용되지 않습니다). 환경의 setuptools 를 쓰도록
# --no-build-isolation 으로 먼저 깔아 둡니다.
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth 1 --branch ${ISAACLAB_REF} https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH} \
 && source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} \
 && { python -c "import pkg_resources" 2>/dev/null || pip install "setuptools<82"; } \
 && pip install --no-build-isolation "flatdict==4.0.1" \
 && cd ${ISAACLAB_PATH} && ./isaaclab.sh -i \
 && python -c "import importlib.metadata as m; import isaaclab; from isaaclab.app import AppLauncher; print('isaaclab', [m.version(d) for d in ['isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl','isaaclab_mimic']])"

# isaaclab.sh -i 가 rl_games / rsl_rl / sb3 / skrl 을 깔면서 torch 를 갈아치우는 일이 있어
# 마지막에 핀으로 되돌리고 검증합니다.
RUN --mount=type=cache,target=/root/.cache/pip \
    source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} \
 && pip install -U "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url ${TORCH_INDEX} \
 && python -c "import sys, torch, isaacsim; \
print('python     :', sys.version.split()[0]); \
print('torch      :', torch.__version__); \
print('torch cuda :', torch.version.cuda); \
assert sys.version_info[:2] == (3, 11), sys.version; \
assert torch.__version__.startswith('${TORCH_VERSION}'), torch.__version__; \
assert torch.version.cuda == '12.8', torch.version.cuda"
