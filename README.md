# gpu-build (uv + PyTorch CPU / CUDA 12.8)

Streamlit으로 라이브러리 버전 확인 후, scikit-learn → PyTorch(FashionMNIST) → YOLOv8 순으로 동작을 검증하는 프로젝트입니다.

**README가 바뀐 이유:** 예전 내용은 GitHub 저장소를 처음 만들 때 쓰는 `git init` / `remote` 예시뿐이었습니다. 지금 저장소는 **uv로 의존성을 고정**하고, **PyTorch만 CPU용과 CUDA 12.8용 중 하나**를 고르는 구조(`pyproject.toml`의 optional extra)라서, 실제로 따라 할 수 있는 설치·실행 절차로 다시 정리했습니다.

---

## 사전 요건

- **Python 3.12** (이 프로젝트는 `>=3.12,<3.13`만 허용)
- GPU(cu128)를 쓸 경우: NVIDIA 드라이버가 설치되어 있고, **CUDA 12.8**용 PyTorch 휠과 맞는 환경인지 확인하세요.

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

## 3. 의존성 설치 — CPU **또는** GPU 중 하나만

`pyproject.toml`에서 `cpu`와 `cu128` extra는 **동시에 쓸 수 없습니다** (`tool.uv.conflicts`). 둘 중 **하나만** 골라 `uv sync` 하세요.

### CPU 전용 PyTorch

```bash
uv sync --extra cpu
```

### CUDA 12.8 PyTorch (GPU)

```bash
uv sync --extra cu128
```

- 처음 받을 때 시간이 걸릴 수 있습니다.
- **다른 쪽으로 바꾸려면** 위 명령으로 다시 `uv sync` 하면 됩니다(락 파일과 환경이 해당 extra 기준으로 맞춰집니다).

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
# 또는 GPU
uv sync --extra cu128 --python 3.12
```

### `cpu`와 `cu128`를 동시에 넣었다 / 충돌한다

한 번에 **하나의 extra만** 지정해야 합니다. 이미 잘못 맞춰진 경우, 원하는 쪽 한 줄로 다시 동기화합니다.

```bash
uv sync --extra cpu
# 또는
uv sync --extra cu128
```

### 동기화는 됐는데 PyTorch·CUDA가 이상할 때

가상환경을 유지한 채 패키지를 다시 깔아 봅니다.

```bash
uv sync --reinstall --extra cpu
# 또는
uv sync --reinstall --extra cu128
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

삭제 후 다시 `uv sync --extra cpu` 또는 `uv sync --extra cu128`를 실행합니다.

### 캐시·다운로드가 꼬였을 때

```bash
uv cache clean
```

이후 다시:

```bash
uv sync --extra cpu
# 또는
uv sync --extra cu128
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

### GPU(cu128)인데 `cuda: False`일 때

- NVIDIA 드라이버·GPU가 인식되는지 OS에서 먼저 확인합니다.
- **CPU 전용**으로 `uv sync --extra cpu`만 한 상태이면 CUDA가 비활성인 것이 정상입니다. GPU를 쓰려면 `uv sync --extra cu128`로 맞춘 뒤 위 `torch.cuda.is_available()`를 다시 확인합니다.

---

## 6. 참고

- `models/`, `data/`는 `.gitignore`에 포함되어 있으며, 실행 시 데이터/가중치가 여기에 쌓일 수 있습니다.
- PyTorch 인덱스 설정은 [uv + PyTorch 가이드](https://docs.astral.sh/uv/guides/integration/pytorch/#configuring-accelerators-with-optional-dependencies)와 동일한 패턴입니다.
