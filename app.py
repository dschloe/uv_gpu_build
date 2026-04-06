"""Streamlit: library versions, then sklearn → PyTorch → YOLOv8 build + test (notebook parity)."""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _models_dir() -> Path:
    d = PROJECT_ROOT / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def show_library_versions() -> None:
    from utils.env_check import print_library_versions

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_library_versions()
    st.code(buf.getvalue().strip(), language="text")


def build_and_test_sklearn(models: Path) -> None:
    import joblib
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    try:
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data,
            iris.target,
            test_size=0.25,
            random_state=42,
            stratify=iris.target,
        )
        clf = RandomForestClassifier(n_estimators=30, random_state=42)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        sk_path = models / "sklearn_iris_rf.joblib"
        joblib.dump(clf, sk_path)
        st.success(f"Scikit-learn: 학습·저장 완료 ({sk_path.name}). Holdout accuracy: {acc:.4f}")
        st.dataframe({"True": y_test[:10], "Pred": pred[:10]})
    except Exception as exc:
        st.error(f"Scikit-learn 단계 실패: {type(exc).__name__}: {exc}")


def build_and_test_pytorch(models: Path) -> None:
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms

    from utils.env_check import configure_korean_font_matplotlib, resolve_torch_device

    class SimpleCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(26 * 26 * 32, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu(self.conv1(x))
            x = x.view(x.size(0), -1)
            return self.fc1(x)

    try:
        try:
            _, device = resolve_torch_device()
        except Exception:
            device = torch.device("cpu")

        transform = transforms.Compose([transforms.ToTensor()])
        data_root = str(PROJECT_ROOT / "data")
        train_dataset = datasets.FashionMNIST(data_root, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            try:
                data, target = data.to(device), target.to(device)
            except Exception:
                device = torch.device("cpu")
                data, target = data.to(device), target.to(device)
                model.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx >= 2:
                break

        model_path = models / "fashionmnist_cnn.pt"
        torch.save(model.state_dict(), model_path)
        st.success(f"PyTorch: 짧은 학습 후 저장 완료 ({model_path.name}), device={device}")

        # --- test on 8 samples (same as previous app) ---
        model.eval()
        test_dataset = datasets.FashionMNIST(data_root, train=False, download=True, transform=transform)
        images, labels = [], []
        for idx in range(8):
            img, label = test_dataset[idx]
            images.append(img)
            labels.append(label)
        images_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            outputs = model(images_tensor)
            preds = outputs.argmax(dim=1).cpu().numpy()

        classes = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ]
        configure_korean_font_matplotlib()
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i].squeeze(0).cpu().numpy(), cmap="gray")
            ax.set_title(f"정답: {classes[labels[i]]}\n예측: {classes[preds[i]]}")
            ax.axis("off")
        plt.tight_layout()
        out_buf = io.BytesIO()
        plt.savefig(out_buf, format="png")
        plt.close(fig)
        out_buf.seek(0)
        st.image(out_buf, caption="FashionMNIST 샘플 예측", width="stretch")
    except Exception as exc:
        st.error(f"PyTorch 단계 실패: {type(exc).__name__}: {exc}")


def build_and_test_yolov8(models: Path) -> None:
    import cv2
    import torch
    from sklearn.datasets import load_sample_images
    from ultralytics import YOLO

    from utils.env_check import resolve_torch_device

    try:
        _, device = resolve_torch_device()
    except Exception:
        device = torch.device("cpu")

    target = models / "yolov8n.pt"
    root_weights = PROJECT_ROOT / "yolov8n.pt"
    try:
        # 가중치는 항상 models/yolov8n.pt 한 곳만 사용
        if root_weights.is_file():
            if not target.is_file():
                shutil.move(str(root_weights), target)
                st.info(f"YOLO: 프로젝트 루트의 가중치를 `{target.relative_to(PROJECT_ROOT)}`(으)로 옮겼습니다.")
            else:
                try:
                    root_weights.unlink()
                except OSError:
                    pass

        if not target.is_file():
            # Ultralytics는 cwd에 받으므로 models/에서 받아 경로를 고정
            prev = os.getcwd()
            try:
                os.chdir(models)
                model = YOLO("yolov8n.pt")
            finally:
                os.chdir(prev)
            st.info(f"YOLO: 가중치를 `{target.relative_to(PROJECT_ROOT)}`에 저장했습니다.")
        else:
            model = YOLO(str(target))
            st.info(f"YOLO: 기존 가중치 로드 — {target.name}")

        imgs = load_sample_images()
        rgb = imgs.images[1]
        results = model.predict(source=rgb, device=device, verbose=False)
        r0 = results[0]
        nbox = len(r0.boxes) if r0.boxes is not None else 0
        st.success(f"YOLOv8: 추론 완료 (검출 박스 수: {nbox}), device={device}")

        plotted = r0.plot()
        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
        st.image(plotted_rgb, caption="YOLO 예측 (sklearn 내장 꽃 이미지)", width="stretch")

        manifest = {
            "sklearn_model": str((models / "sklearn_iris_rf.joblib").resolve()),
            "yolo_weights": str(target.resolve()),
            "torch_device_hint": str(device),
        }
        mp = models / "deploy_manifest.json"
        mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        st.caption(f"manifest 저장: {mp.name}")
    except Exception as exc:
        st.error(f"YOLOv8 단계 실패: {type(exc).__name__}: {exc}")


def main() -> None:
    os.chdir(PROJECT_ROOT)
    st.set_page_config(page_title="설치/모델 검증", layout="wide")
    st.title("Python ML/AI 환경 — 버전 확인 및 모델 순차 생성·테스트")

    st.header("1. 주요 라이브러리 버전")
    show_library_versions()

    models = _models_dir()

    st.header("2. Scikit-learn — 모델 생성 및 바로 테스트")
    build_and_test_sklearn(models)

    st.header("3. PyTorch (FashionMNIST) — 모델 생성 및 바로 테스트")
    build_and_test_pytorch(models)

    st.header("4. YOLOv8 (Ultralytics) — 가중치 준비 및 바로 테스트")
    build_and_test_yolov8(models)

    st.markdown("---")
    st.caption("`models/`, `data/`는 .gitignore에 포함되어 있습니다.")


if __name__ == "__main__":
    main()
