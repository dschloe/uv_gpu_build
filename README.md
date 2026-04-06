# gpu-build (uv + PyTorch CPU / CUDA 휠 선택)

Streamlit으로 라이브러리 버전 확인 후, scikit-learn → PyTorch(FashionMNIST) → YOLOv8 순으로 동작을 검증하는 프로젝트입니다.

**README가 바뀐 이유:** 예전 내용은 GitHub 저장소를 처음 만들 때 쓰는 `git init` / `remote` 예시뿐이었습니다. 지금 저장소는 **uv로 의존성을 고정**하고, **PyTorch는 CPU용과 GPU용 CUDA 휠(cu118 / cu128) 중 하나**를 고르는 구조(`pyproject.toml`의 optional extra)라서, 실제로 따라 할 수 있는 설치·실행 절차로 다시 정리했습니다.

---

## 사전 요건

- **Python 3.12** (이 프로젝트는 `>=3.12,<3.13`만 허용)
- **GPU 사용 시:** NVIDIA 드라이버가 설치되어 있어야 하며, 설치하는 PyTorch 휠의 CUDA 태그(cu118 / cu128)와 호환되는지 확인하세요. (세부 버전·조합은 아래 [CUDA / PyTorch 버전은 어디서 보나요?](#cuda--pytorch-버전은-어디서-보나요) 참고)

---

## 1. uv 설치

공식 문서: [Installing uv](https://docs.astral.sh/uv/getting-started/installation/)

**Windows (PowerShell, 권장)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

설치 후 터미널을 다시 열고 확인:

```powershell
uv --version
```

**기타:** macOS/Linux는 위 문서의 curl 설치 스크립트 또는 `pip install uv` 등을 사용할 수 있습니다.

---

## 2. 저장소 받기 & 프로젝트 폴더로 이동

```bash
git clone https://github.com/dschloe/uv_gpu_build.git
cd uv_gpu_build
```

---

## 3. 의존성 설치 — CPU **또는** GPU(cu118 / cu128) **하나만**

`pyproject.toml`에서 `cpu`, `cu118`, `cu128` extra는 **동시에 쓸 수 없습니다** (`tool.uv.conflicts`). **하나만** 골라 `uv sync` 하세요.

### CPU 전용 PyTorch

```bash
uv sync --extra cpu
```

### GPU — CUDA 11.8 휠 (노트북·호환 우선)

저사양·노트북 GPU처럼 **최신 CUDA 12.8 휠보다 예전 런타임에 맞추고 싶을 때** 먼저 시도해 볼 수 있는 선택지입니다.

```bash
uv sync --extra cu118
```

### GPU — CUDA 12.8 휠

비교적 최신 NVIDIA 드라이버와 함께 쓰기 좋은 조합입니다.

```bash
uv sync --extra cu128
```

- 처음 받을 때 시간이 걸릴 수 있습니다.
- **다른 extra로 바꾸려면** 원하는 쪽 한 줄로 다시 `uv sync` 하면 됩니다(락 파일과 환경이 해당 extra 기준으로 맞춰집니다).

---

## 4. 앱 실행 (Streamlit)

프로젝트 루트에서:

```bash
uv run streamlit run app.py
```

브라우저에서 열리는 페이지에서 순서대로 버전 출력 → sklearn → PyTorch → YOLO 흐름을 확인할 수 있습니다.

---

## 5. 설치 문제 대처 (uv 명령어 위주)

아래는 **의존성 설치·동기화**에서 자주 나오는 경우에 대한 대응입니다. 가능하면 프로젝트 루트에서 실행하세요.

### Python 버전이 안 맞을 때

이 프로젝트는 **3.12**만 허용합니다. uv로 3.12를 깔고 맞춘 뒤 다시 동기화합니다.

```bash
uv python install 3.12
uv sync --extra cpu --python 3.12
# 또는 GPU (예: cu118)
uv sync --extra cu118 --python 3.12
```

### extra를 여러 개 넣었다 / 충돌한다

한 번에 **하나의 extra만** 지정해야 합니다. 이미 잘못 맞춰진 경우, 원하는 쪽 한 줄로 다시 동기화합니다.

```bash
uv sync --extra cpu
# 또는
uv sync --extra cu118
# 또는
uv sync --extra cu128
```

### 동기화는 됐는데 PyTorch·CUDA가 이상할 때

가상환경을 유지한 채 패키지를 다시 깔아 봅니다.

```bash
uv sync --reinstall --extra cpu
# 또는 (예)
uv sync --reinstall --extra cu118
```

그래도 안 되면 **`.venv` 폴더를 삭제**한 뒤, 같은 `uv sync --extra …`를 처음부터 다시 실행합니다. 아래는 **프로젝트 루트**에서 실행하는 예입니다.

**PowerShell**

```powershell
Remove-Item -Recurse -Force .venv
```

**CMD**

```cmd
rmdir /s /q .venv
```

**Git Bash**

```bash
rm -rf .venv
```

삭제 후 다시 `uv sync --extra cpu` 또는 원하는 GPU extra(`cu118` / `cu128`)로 실행합니다.

### 캐시·다운로드가 꼬였을 때

```bash
uv cache clean
```

이후 다시:

```bash
uv sync --extra cpu
# 또는 GPU extra 하나
uv sync --extra cu118
```

### 원인 파악을 위해 로그·환경 확인

동기화 로그를 자세히 보려면:

```bash
uv sync --extra cpu -v
```

설치된 `torch`와 CUDA 사용 가능 여부는 실행 환경에서 확인합니다.

```bash
uv run python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

의존성 트리:

```bash
uv tree
```

### GPU extra인데 `cuda: False`일 때

- NVIDIA 드라이버·GPU가 인식되는지 OS에서 먼저 확인합니다.
- **CPU 전용**으로 `uv sync --extra cpu`만 한 상태이면 CUDA가 비활성인 것이 정상입니다. GPU를 쓰려면 `cu118` 또는 `cu128` 중 하나로 맞춘 뒤 위 `torch.cuda.is_available()`를 다시 확인합니다.
- 같은 GPU라도 **다른 CUDA 휠**이 맞을 수 있으니, 안 되면 `cu118` ↔ `cu128`를 바꿔 보세요(둘 다 동시에는 안 됨).

---

## 6. 참고

### 내 PC에서 CUDA(관련) 버전 확인

아래는 **어떤 숫자가 무엇을 뜻하는지**가 서로 다릅니다. 헷갈리면 `nvidia-smi`와 PyTorch 출력을 같이 보면 됩니다.

**1) NVIDIA 드라이버가 인식되는지 + 드라이버가 지원하는 CUDA 상한**

PowerShell, CMD, Git Bash 어디서든(드라이버·`nvidia-smi`가 설치된 경우):

```bash
nvidia-smi
```

출력 **오른쪽 위**에 보이는 `CUDA Version`은 **이 드라이버가 호환하는 CUDA 런타임 상한**에 가깝고, PC에 **CUDA 툴킷이 깔려 있다는 뜻은 아닙니다.**

같은 줄에 **13.1처럼 보이는 숫자**가 나와도, 이 프로젝트의 PyTorch extra 이름(**`cu118`**, **`cu128`**)과 **숫자가 일치할 필요는 없습니다.** PyTorch는 선택한 extra에 맞는 휠을 받아 오고, 그 안에 맞는 CUDA 런타임이 들어 있으며, 드라이버가 충분히 새면 보통 함께 동작합니다.

**2) CUDA 툴킷을 따로 설치한 경우 (컴파일러)**

```bash
nvcc --version
```

`nvcc`가 없다고 나오면 툴킷이 없는 것이며, **PyTorch를 휠로만 쓰는 경우에는 필수는 아닙니다.**

**3) 이미 이 프로젝트로 PyTorch를 설치한 뒤 — 휠에 포함된 CUDA**

```bash
uv run python -c "import torch; print('torch:', torch.__version__); print('torch CUDA build:', torch.version.cuda)"
```

여기서 `torch.version.cuda`는 **지금 설치된 PyTorch 바이너리가 빌드된 CUDA 버전**(예: `11.8`, `12.8`)입니다. `cu118` / `cu128` extra와 대응하는지 확인할 때 쓰면 됩니다.

---

### CUDA / PyTorch 버전은 어디서 보나요?

이 README에는 **cu118·cu128 두 가지 GPU 휠**만 미리 넣어 두었습니다. **드라이버·GPU와 어떤 CUDA 빌드가 맞는지**는 아래에서 확인하는 것이 좋습니다.

- **PyTorch 공식 설치 선택 화면 (CUDA 버전·명령 예시):** [Get Started](https://pytorch.org/get-started/locally/)
- **PyTorch 휠 인덱스(폴더 구조 확인용):** [download.pytorch.org/whl](https://download.pytorch.org/whl/)
- **uv에서 PyTorch·인덱스(extra) 묶는 방법:** [uv — PyTorch accelerators](https://docs.astral.sh/uv/guides/integration/pytorch/#configuring-accelerators-with-optional-dependencies)

다른 `cu…` 조합을 쓰려면 위 문서를 본 뒤 `pyproject.toml`의 optional extra·`[[tool.uv.index]]`·`tool.uv.sources`를 같은 패턴으로 추가하면 됩니다.

`cu118`과 `cu128`은 **서로 다른 PyTorch 인덱스**를 쓰므로, uv가 잡는 **`torch` / `torchvision` 버전이 extra마다 달라질 수 있습니다**(공식 휠에 올라온 조합에 따름).

### 기타

- `models/`, `data/`는 `.gitignore`에 포함되어 있으며, 실행 시 데이터/가중치가 여기에 쌓일 수 있습니다.
