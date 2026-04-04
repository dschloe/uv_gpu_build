Windows에서 PowerShell로 uv 설치하기
=================================

Windows 환경에서 PowerShell을 사용하여 [uv](https://github.com/astral-sh/uv)를 설치하려면 다음 방법을 따라주세요.

### 1. scoop을 활용한 설치 (권장)

1. **scoop이 설치되어 있지 않다면, 아래 명령어로 scoop 설치:**
    ```powershell
    Set-ExecutionPolicy RemoteSigned -scope CurrentUser
    irm get.scoop.sh | iex
    ```

2. **uv 설치:**
    ```powershell
    scoop install uv
    ```

### 2. winget을 활용한 설치

Windows 10 이상에서는 `winget` 패키지 관리자를 사용할 수 있습니다.

```powershell
winget install astral.sh.uv
```

### 3. 수동 설치

1. [uv Releases 페이지](https://github.com/astral-sh/uv/releases)에서 
   `uv-x86_64-pc-windows-msvc.zip` 혹은 비슷한 최신 Windows용 패키지 zip 파일을 다운로드합니다.

2. 다운로드한 zip 파일의 압축을 풉니다.

3. 원하는 폴더(예: `C:\Program Files\uv\`)에 `uv.exe`를 복사합니다.

4. 해당 폴더를 환경 변수 `Path`에 추가합니다:
   
   ```powershell
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\uv", [EnvironmentVariableTarget]::Machine)
   ```

### 4. 설치 확인

PowerShell에서 다음 명령어를 입력하여 uv가 제대로 설치되었는지 확인합니다.
```powershell
uv --version
```

> 더 구체적인 안내는 [공식 설치 문서](https://docs.astral.sh/uv/getting-started/installation/)를 참조하세요.


---

## uv sync 명령어란?

`uv sync` 명령어는 Python 프로젝트의 의존성 관리를 단순화하기 위해 사용됩니다.  
프로젝트 디렉토리에 `pyproject.toml` 또는 `requirements.txt` 파일이 있을 때, 이 파일에 정의된 패키지 버전과 실제 가상환경(venv) 또는 시스템에 설치된 패키지의 버전을 동기화(synchronize)합니다.

즉, `uv sync`를 실행하면 지정된 패키지들이 프로젝트에 맞게 설치/업데이트/삭제되어 의존성이 정확하게 맞춰집니다.

### 주요 기능

- `pyproject.toml`, `requirements.txt` 등 선언된 의존성 목록과 실제 설치된 패키지를 일치시킴
- 누락된 패키지는 설치, 불필요한 패키지는 제거
- 빠르고 신뢰성 있게 동기화 진행

---

## 사용 예시

**(최초 한 번만 아래 명령어로 uv 캐시를 비워주세요 ‑ 해결되지 않는 패키지 충돌/이슈 예방용)**
```powershell
Remove-Item -Recurse -Force .\.venv
uv cache clean
Remove-Item -Force .\uv.lock
```

1. **현재 디렉토리에 `pyproject.toml` 파일이 있는 경우:**

    # (CPU 또는 CUDA 중 하나를 선택하여 실행)
    # CPU 환경 설치:
    uv sync --extra cpu

    # 또는 CUDA(높은 성능 GPU 활용) 환경 설치:
    uv sync --extra cu128

    또는 `requirements.txt` 파일 기준으로 동기화하려면,
    ```powershell
    uv sync -r requirements.txt
    ```

2. **추가 예시 (`--system` 옵션):**
   
   시스템 전체에 패키지를 동기화하려면 아래와 같이 실행할 수 있습니다.
    ```powershell
    uv sync --system
    ```

> 상세 사용법은 [uv 공식 문서의 sync 명령어](https://docs.astral.sh/uv/reference/cli/#sync)도 참고하세요.


### uv run을 사용해서 특정 `.py` 파일 순차 실행하기

이미 가상환경이 만들어져 있는 상태라면, `uv run` 명령어를 사용하여 해당 가상환경에서 Python 파일을 실행할 수 있습니다.  
예를 들어, 프로젝트 디렉토리에 있는 `lecture_install_verify_all.py` 파일을 먼저 실행하고, 그 다음 `streamlit_verify.py` 파일을 실행하려면 아래와 같이 명령어를 입력합니다.

#### PowerShell/Bash 공통

```bash
uv run python notebooks/lecture_install_verify_all.py
uv run python streamlit_verify.py
```

이 명령어는 uv가 현재 디렉토리에 존재하는 `.venv`(또는 표준 Python 가상환경 폴더)를 감지해, 그 환경에서 파이썬 스크립트를 실행합니다.  
즉, 별도의 가상환경 활성화 과정 없이 uv가 알아서 해당 환경을 사용합니다.

> 참고: `uv run`는 단일 명령 실행용이며, 여러 파일을 순차적으로 실행할 땐 위 예시처럼 한 줄씩 실행하면 됩니다.

자세한 내용은 [uv 공식 문서의 run 명령어](https://docs.astral.sh/uv/reference/cli/#run)를 참고하세요.