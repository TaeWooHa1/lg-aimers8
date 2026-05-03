# ============================================================
# LG Aimers 대회 서버 환경 재현 Dockerfile
# - OS: Ubuntu 22.04
# - Python: 3.11
# - CUDA: 12.8
# - 패키지: 대회 서버 고정 버전 + lm-eval
# ============================================================

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# 환경 변수
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── 시스템 패키지 (대회 서버 목록과 동일) ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    git \
    git-lfs \
    ninja-build \
    libomp-dev \
    libblas3 \
    liblapack3 \
    gfortran \
    libatlas-base-dev \
    curl \
    wget \
    ca-certificates \
    unzip \
    procps \
    cmake \
    tzdata \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python 3.11 설치 (deadsnakes PPA) ──
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-distutils \
    python3.11-dev \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# python3 -> python3.11 로 설정
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# pip 설치
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# ── Python 패키지 설치 ──
WORKDIR /app

COPY requirements_server.txt .

# torch는 PyTorch 공식 cu128 인덱스에서 설치
RUN pip install --no-cache-dir \
    torch==2.9.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# 나머지 패키지 설치 (torch 제외)
RUN grep -v "^torch==" requirements_server.txt | \
    grep -v "^#" | \
    grep -v "^$" | \
    pip install --no-cache-dir -r /dev/stdin

# ── 작업 디렉토리 ──
# evaluate_local.py와 모델은 docker-compose의 volumes로 마운트됩니다
WORKDIR /workspace

CMD ["bash"]
