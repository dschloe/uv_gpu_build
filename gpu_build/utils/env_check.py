"""Cross-platform checks for local desktop (Windows, macOS/M1, Linux)."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from typing import Optional


def print_library_versions(module_names: Optional[list[str]] = None) -> None:
    """Print versions for the teaching stack. Uses try/except per package."""
    if module_names is None:
        module_names = [
            "numpy",
            "pandas",
            "pyarrow",
            "scipy",
            "sklearn",
            "matplotlib",
            "seaborn",
            "plotly",
            "PIL",
            "cv2",
            "torch",
            "torchvision",
            "ultralytics",
            "streamlit",
            "joblib",
            "openpyxl",
            "xgboost",
            "lightgbm",
            "catboost",
            "optuna",
            "statsmodels",
            "umap",
            "tqdm",
            "nbformat",
            "jupyterlab",
            "ucimlrepo",
        ]

    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("---")

    for name in module_names:
        label = name
        try:
            if name == "sklearn":
                mod = importlib.import_module("sklearn")
                ver = getattr(mod, "__version__", importlib.metadata.version("scikit-learn"))
                label = "scikit-learn"
            elif name == "cv2":
                mod = importlib.import_module("cv2")
                ver = getattr(mod, "__version__", "?")
                label = "opencv-python"
            elif name == "PIL":
                mod = importlib.import_module("PIL")
                ver = getattr(mod, "__version__", importlib.metadata.version("pillow"))
                label = "pillow"
            elif name == "joblib":
                mod = importlib.import_module("joblib")
                ver = getattr(mod, "__version__", importlib.metadata.version("joblib"))
            elif name == "umap":
                mod = importlib.import_module("umap")
                ver = getattr(mod, "__version__", importlib.metadata.version("umap-learn"))
                label = "umap-learn"
            else:
                mod = importlib.import_module(name)
                ver = getattr(mod, "__version__", None)
                if ver is None:
                    dist_map = {
                        "torchvision": "torchvision",
                        "PIL": "pillow",
                        "cv2": "opencv-python",
                        "sklearn": "scikit-learn",
                        "umap": "umap-learn",
                    }
                    dist_name = dist_map.get(name, name)
                    ver = importlib.metadata.version(dist_name)
            print(f"{label}: {ver}")
        except Exception as exc:
            print(f"{label}: ERROR ({type(exc).__name__}) {exc}")


def configure_korean_font_matplotlib() -> str:
    """Set matplotlib/seaborn font for Korean labels. Safe on all OS."""
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    import seaborn as sns

    candidates = [
        "Malgun Gothic",
        "Apple SD Gothic Neo",
        "AppleGothic",
        "Nanum Gothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]

    chosen = None
    try:
        try:
            installed = set(fm.get_font_names())
        except AttributeError:
            installed = {f.name for f in fm.fontManager.ttflist}

        for c in candidates:
            if c in installed:
                chosen = c
                break

        if chosen:
            plt.rcParams["font.family"] = chosen
        else:
            plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["axes.unicode_minus"] = False
        sns.set_theme(
            font=plt.rcParams["font.family"],
            rc={"axes.unicode_minus": False},
        )
        return chosen or "DejaVu Sans (Korean font not found, using fallback)"
    except Exception as exc:
        try:
            plt.rcParams["font.family"] = "DejaVu Sans"
            plt.rcParams["axes.unicode_minus"] = False
            sns.set_theme(font="DejaVu Sans", rc={"axes.unicode_minus": False})
        except Exception:
            pass
        return f"DejaVu Sans (font setup error: {exc})"


def resolve_torch_device():
    """Return (label, torch.device) with CUDA > MPS > CPU and try/except."""
    import torch

    try:
        if torch.cuda.is_available():
            return "cuda", torch.device("cuda")
    except Exception:
        pass

    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", torch.device("mps")
    except Exception:
        pass

    return "cpu", torch.device("cpu")
