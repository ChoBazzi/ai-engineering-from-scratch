# AI를 위한 Docker

> 컨테이너는 "내 컴퓨터에서는 되는데"라는 말을 과거의 일로 만든다.

**Type:** Build
**Languages:** Docker
**Prerequisites:** Phase 0, Lessons 01 and 03
**Time:** ~60 minutes

## 학습 목표

- Dockerfile에서 CUDA, PyTorch, AI 라이브러리를 포함한 GPU 지원 Docker 이미지를 빌드한다
- 호스트 디렉터리를 볼륨으로 마운트해 컨테이너를 다시 빌드해도 모델, 데이터셋, 코드가 유지되게 한다
- NVIDIA Container Toolkit을 구성해 컨테이너 내부에서 GPU를 노출한다
- Docker Compose로 여러 서비스로 구성된 AI 애플리케이션(추론 서버 + 벡터 데이터베이스)을 오케스트레이션한다

## 문제

노트북에서 PyTorch 2.3, CUDA 12.4, Python 3.12로 모델을 학습했다. 동료는 PyTorch 2.1, CUDA 11.8, Python 3.10을 사용한다. 동료의 컴퓨터에서는 모델이 충돌한다. Dockerfile은 두 환경 모두에서 동작한다.

AI 프로젝트는 의존성 악몽이 되기 쉽다. 일반적인 스택에는 Python, PyTorch, CUDA 드라이버, cuDNN, 시스템 수준 C 라이브러리, 정확한 컴파일러 버전이 필요한 flash-attn 같은 특수 패키지가 포함된다. Docker는 이 모든 것을 어디서나 동일하게 실행되는 하나의 이미지로 패키징한다.

## 개념

Docker는 코드, 런타임, 라이브러리, 시스템 도구를 컨테이너라는 격리된 단위로 감싼다. 가벼운 가상 머신이라고 생각하면 된다. 다만 자체 OS 커널을 실행하는 대신 호스트 OS 커널을 공유하므로 몇 분이 아니라 몇 초 만에 시작된다.

```mermaid
graph TD
    subgraph without["Without Docker"]
        A1["Your machine<br/>Python 3.12<br/>CUDA 12.4<br/>PyTorch 2.3"] -->|crashes| X1["???"]
        A2["Their machine<br/>Python 3.10<br/>CUDA 11.8<br/>PyTorch 2.1"] -->|crashes| X2["???"]
        A3["Server<br/>Python 3.11<br/>CUDA 12.1<br/>PyTorch 2.2"] -->|crashes| X3["???"]
    end

    subgraph with_docker["With Docker — Same image everywhere"]
        B1["Your machine<br/>Python 3.12 | CUDA 12.4<br/>PyTorch 2.3 | Your code"]
        B2["Their machine<br/>Python 3.12 | CUDA 12.4<br/>PyTorch 2.3 | Your code"]
        B3["Server<br/>Python 3.12 | CUDA 12.4<br/>PyTorch 2.3 | Your code"]
    end
```

### AI 프로젝트에 Docker가 특히 더 필요한 이유

1. **GPU 드라이버는 취약하다.** CUDA 12.4 코드는 CUDA 11.8에서 실행되지 않는다. Docker는 컨테이너 내부의 CUDA 툴킷을 격리하면서 NVIDIA Container Toolkit을 통해 호스트 GPU 드라이버를 공유한다.

2. **모델 가중치는 크다.** 7B 파라미터 모델은 fp16에서 14 GB다. 다시 빌드할 때마다 다운로드하고 싶지는 않을 것이다. Docker 볼륨을 사용하면 호스트의 models 디렉터리를 마운트할 수 있다.

3. **다중 서비스 아키텍처가 흔하다.** 실제 AI 애플리케이션은 Python 스크립트 하나가 아니다. 추론 서버, RAG용 벡터 데이터베이스, 어쩌면 웹 프런트엔드까지 포함된다. Docker Compose는 이 모든 것을 명령 하나로 오케스트레이션한다.

### 핵심 어휘

| 용어 | 의미 |
|------|------|
| Image | 읽기 전용 템플릿. 레시피. Dockerfile에서 빌드된다. |
| Container | 이미지의 실행 중인 인스턴스. 주방. |
| Dockerfile | 이미지를 빌드하기 위한 지시문. 레이어 단위로 구성된다. |
| Volume | 컨테이너를 다시 시작해도 유지되는 영구 스토리지. |
| docker-compose | YAML로 다중 컨테이너 애플리케이션을 정의하는 도구. |

### AI에서 흔한 컨테이너 패턴

```
Dev Container
  Full toolkit. Editor support. Jupyter. Debugging tools.
  Used during development and experimentation.

Training Container
  Minimal. Just the training script and dependencies.
  Runs on GPU clusters. No editor, no Jupyter.

Inference Container
  Optimized for serving. Small image. Fast cold start.
  Runs behind a load balancer in production.
```

## 직접 만들기

### Step 1: Docker 설치

```bash
# macOS
brew install --cask docker
open /Applications/Docker.app

# Ubuntu
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in for group change to take effect
```

확인:

```bash
docker --version
docker run hello-world
```

### Step 2: NVIDIA Container Toolkit 설치(Linux + NVIDIA GPU)

이 도구를 사용하면 Docker 컨테이너가 GPU에 접근할 수 있다. macOS와 Windows(WSL2) 사용자는 건너뛰어도 된다. 해당 플랫폼에서는 Docker Desktop이 GPU passthrough를 다르게 처리한다.

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

컨테이너 내부에서 GPU 접근을 테스트한다:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

GPU 정보가 보이면 툴킷이 동작하는 것이다.

### Step 3: 베이스 이미지 이해하기

올바른 베이스 이미지를 고르면 디버깅 시간을 몇 시간 줄일 수 있다.

```
nvidia/cuda:12.4.1-devel-ubuntu22.04
  Full CUDA toolkit. Compilers included.
  Use for: building packages that need nvcc (flash-attn, bitsandbytes)
  Size: ~4 GB

nvidia/cuda:12.4.1-runtime-ubuntu22.04
  CUDA runtime only. No compilers.
  Use for: running pre-built code
  Size: ~1.5 GB

pytorch/pytorch:2.3.1-cuda12.4-cudnn9-runtime
  PyTorch pre-installed on top of CUDA.
  Use for: skipping the PyTorch install step
  Size: ~6 GB

python:3.12-slim
  No CUDA. CPU only.
  Use for: inference on CPU, lightweight tools
  Size: ~150 MB
```

### Step 4: AI 개발용 Dockerfile 작성

`code/Dockerfile`에 있는 Dockerfile이다. 차례대로 살펴보자:

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN python -m pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    jupyter \
    transformers \
    datasets \
    accelerate \
    safetensors

WORKDIR /workspace

VOLUME ["/workspace", "/models"]

EXPOSE 8888

CMD ["python"]
```

빌드한다:

```bash
docker build -t ai-dev -f phases/00-setup-and-tooling/07-docker-for-ai/code/Dockerfile .
```

처음에는 시간이 걸린다(CUDA 베이스 이미지 + PyTorch 다운로드). 이후 빌드는 캐시된 레이어를 사용한다.

실행한다:

```bash
docker run --rm -it --gpus all \
    -v $(pwd):/workspace \
    -v ~/models:/models \
    ai-dev python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

컨테이너 안에서 Jupyter를 실행한다:

```bash
docker run --rm -it --gpus all \
    -v $(pwd):/workspace \
    -v ~/models:/models \
    -p 8888:8888 \
    ai-dev jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Step 5: 데이터와 모델을 위한 볼륨 마운트

볼륨 마운트는 AI 작업에서 매우 중요하다. 볼륨이 없으면 컨테이너가 멈출 때 14 GB 모델 다운로드도 사라진다.

```bash
# Mount your code
-v $(pwd):/workspace

# Mount a shared models directory
-v ~/models:/models

# Mount datasets
-v ~/datasets:/data
```

학습 스크립트 안에서는 마운트된 경로에서 로드한다:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("/models/llama-7b")
```

모델은 호스트 파일시스템에 존재한다. 다시 다운로드하지 않고 컨테이너를 원하는 만큼 다시 빌드할 수 있다.

### Step 6: 다중 서비스 AI 앱을 위한 Docker Compose

실제 RAG 애플리케이션에는 추론 서버와 벡터 데이터베이스가 필요하다. Docker Compose는 두 서비스를 명령 하나로 실행한다.

`code/docker-compose.yml`을 보자:

```yaml
services:
  ai-dev:
    build:
      context: .
      dockerfile: Dockerfile
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ../../../:/workspace
      - ~/models:/models
      - ~/datasets:/data
    ports:
      - "8888:8888"
    stdin_open: true
    tty: true
    command: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

  qdrant:
    image: qdrant/qdrant:v1.12.5
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

모든 것을 시작한다:

```bash
cd phases/00-setup-and-tooling/07-docker-for-ai/code
docker compose up -d
```

이제 AI 개발 컨테이너는 서비스 이름으로 `http://qdrant:6333`의 벡터 데이터베이스에 접근할 수 있다. Docker Compose는 공유 네트워크를 자동으로 만든다.

AI 컨테이너 안에서 연결을 테스트한다:

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="qdrant", port=6333)
print(client.get_collections())
```

모든 것을 중지한다:

```bash
docker compose down
```

qdrant 볼륨도 함께 삭제하려면 `-v`를 추가한다:

```bash
docker compose down -v
```

### Step 7: AI 작업에 유용한 Docker 명령

```bash
# List running containers
docker ps

# List all images and their sizes
docker images

# Remove unused images (reclaim disk space)
docker system prune -a

# Check GPU usage inside a running container
docker exec -it <container_id> nvidia-smi

# Copy a file from container to host
docker cp <container_id>:/workspace/results.csv ./results.csv

# View container logs
docker logs -f <container_id>
```

## 사용하기

이제 재현 가능한 AI 개발 환경이 생겼다. 이 과정의 나머지 부분에서는:

- `docker compose up`으로 개발 환경과 벡터 데이터베이스를 함께 시작한다
- 코드, 모델, 데이터를 볼륨으로 마운트해 다시 빌드해도 아무것도 잃지 않게 한다
- 새 Python 패키지가 필요한 lesson에서는 Dockerfile에 추가하고 다시 빌드한다
- 팀원과 Dockerfile을 공유한다. 팀원도 정확히 같은 환경을 얻게 된다.

### GPU가 없다면?

`--gpus all` 플래그와 NVIDIA deploy 블록을 제거한다. CPU 기반 lesson에서는 컨테이너가 여전히 동작한다. PyTorch는 CUDA가 없음을 감지하고 자동으로 CPU로 fallback한다.

## 연습 문제

1. Dockerfile을 빌드하고 컨테이너 안에서 `python -c "import torch; print(torch.__version__)"`를 실행한다
2. docker-compose 스택을 시작하고 AI 컨테이너에서 `http://qdrant:6333/collections`로 Qdrant에 접근할 수 있는지 확인한다
3. Dockerfile에 `flask`를 추가하고 다시 빌드한 뒤 포트 5000에서 간단한 API 서버를 실행한다. `-p 5000:5000`으로 포트를 매핑한다
4. `docker images`로 이미지 크기를 측정한다. 베이스 이미지를 `devel`에서 `runtime`으로 바꿔 보고 크기를 비교한다

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|------------------|-----------|
| Container | "가벼운 VM" | 호스트 커널을 사용하면서 자체 파일시스템과 네트워크를 가진 격리된 프로세스 |
| Image layer | "캐시된 단계" | 각 Dockerfile 지시문은 레이어를 만든다. 변경되지 않은 레이어는 캐시되므로 재빌드가 빠르다. |
| NVIDIA Container Toolkit | "Docker 안의 GPU" | `--gpus` 플래그를 통해 호스트 GPU를 컨테이너에 노출하는 런타임 hook |
| Volume mount | "공유 폴더" | 호스트의 디렉터리를 컨테이너 안으로 매핑한 것. 컨테이너가 멈춘 뒤에도 변경 사항이 유지된다. |
| Base image | "시작점" | Dockerfile이 기반으로 삼는 `FROM` 이미지. 무엇이 미리 설치되어 있는지를 결정한다. |
