import streamlit as st
import os
import sys
import platform

def set_korean_font_for_matplotlib():
    """OS별 한글폰트를 자동 선택하여 matplotlib에 적용 (Win/Mac/Linux 공용)"""
    try:
        import matplotlib
        import matplotlib.font_manager as fm

        font_names = []
        if sys.platform.startswith('darwin'):
            # macOS: AppleGothic, NanumGothic, Malgun Gothic (in case user installed)
            font_names = ["AppleGothic", "NanumGothic", "Malgun Gothic"]
        elif sys.platform.startswith('win'):
            # Windows: Malgun Gothic, NanumGothic
            font_names = ["Malgun Gothic", "NanumGothic", "AppleGothic"]
        else:
            # Linux: NanumGothic
            font_names = ["NanumGothic", "AppleGothic", "Malgun Gothic"]

        found_font = False
        for fontname in font_names:
            matches = [f for f in fm.findSystemFonts(fontpaths=None, fontext='ttf')
                       if fontname in fm.FontProperties(fname=f).get_name()]
            if matches:
                matplotlib.rc("font", family=fontname)
                found_font = True
                break
        if not found_font:
            # 마지막 대안: 나눔고딕이 미설치 시 Arial로 fallback (한글 미지원)
            matplotlib.rc("font", family="Arial")
        # 폰트 적용 후 한글깨짐 방지
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        pass

def get_library_versions():
    versions = []
    try:
        import numpy
        versions.append(f"numpy: {numpy.__version__}")
    except ImportError:
        versions.append("numpy: not installed")
    try:
        import pandas
        versions.append(f"pandas: {pandas.__version__}")
    except ImportError:
        versions.append("pandas: not installed")
    try:
        import matplotlib
        versions.append(f"matplotlib: {matplotlib.__version__}")
    except ImportError:
        versions.append("matplotlib: not installed")
    try:
        import sklearn
        versions.append(f"scikit-learn: {sklearn.__version__}")
    except ImportError:
        versions.append("scikit-learn: not installed")
    try:
        import torch
        versions.append(f"torch: {torch.__version__}")
    except ImportError:
        versions.append("torch: not installed")
    try:
        import cv2
        versions.append(f"opencv-python (cv2): {cv2.__version__}")
    except ImportError:
        versions.append("opencv-python (cv2): not installed")
    try:
        import ultralytics
        versions.append(f"ultralytics: {ultralytics.__version__}")
    except ImportError:
        versions.append("ultralytics: not installed")
    try:
        import streamlit as _st
        versions.append(f"streamlit: {_st.__version__}")
    except ImportError:
        versions.append("streamlit: not installed")
    try:
        import joblib
        versions.append(f"joblib: {joblib.__version__}")
    except ImportError:
        versions.append("joblib: not installed")
    return versions

def check_library_versions():
    versions = get_library_versions()
    st.code('\n'.join(versions), language='text')

def test_sklearn_model():
    """models/sklearn_iris_rf.joblib 모델 불러와서 iris 예측 테스트"""
    try:
        import joblib
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        model_path = os.path.join("models", "sklearn_iris_rf.joblib")
        if not os.path.isfile(model_path):
            st.warning(f"모델 파일이 존재하지 않습니다: {model_path}")
            return

        model = joblib.load(model_path)
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target
        )
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        st.success(f"Scikit-learn 랜덤포레스트 모델 테스트 성공! Holdout accuracy: {acc:.4f}")
        st.dataframe({
            "True": y_test[:10],
            "Pred": pred[:10],
        })
    except Exception as exc:
        st.error(f"scikit-learn 모델 테스트 실패: {exc}")

def test_yolo_model():
    """models/yolov8n.pt YOLO 모델 불러와서 샘플 이미지 테스트"""
    try:
        from sklearn.datasets import load_sample_images
        import cv2
        from ultralytics import YOLO

        yolo_path = os.path.join("models", "yolov8n.pt")
        if not os.path.isfile(yolo_path):
            st.warning(f"YOLO 모델 파일이 존재하지 않습니다: {yolo_path}")
            return
        images = load_sample_images()
        rgb_sample = images.images[1]

        # OpenCV가 RGB 이미지를 BGR로 변환
        bgr_sample = cv2.cvtColor(rgb_sample, cv2.COLOR_RGB2BGR)
        st.write(f"BGR 샘플 이미지 shape: {bgr_sample.shape}, dtype: {bgr_sample.dtype}")

        model = YOLO(yolo_path)
        # 실습 환경에서 GPU가 없을 수 있으므로 device 파라미터: cpu
        results = model.predict(source=rgb_sample, device="cpu", verbose=False)
        plotted = results[0].plot()
        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
        st.image(plotted_rgb, caption="YOLO 예측 결과 (샘플 꽃 이미지)", width="stretch")
    except Exception as exc:
        st.error(f"YOLO 모델 테스트 실패: {exc}")

def test_fashionmnist_model():
    """
    models/fashionmnist_cnn.pt PyTorch FashionMNIST 모델 불러와서 샘플 이미지 테스트
    CUDA device가 없을 때 map_location을 제대로 사용하는 안전한 로딩.
    """
    try:
        import torch
        import torch.nn as nn
        from torchvision import datasets, transforms

        model_path = os.path.join("models", "fashionmnist_cnn.pt")
        if not os.path.isfile(model_path):
            st.warning(f"FashionMNIST 모델 파일이 존재하지 않습니다: {model_path}")
            return

        # 모델 구조 정의 (notebook과 동일)
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.relu = nn.ReLU()
                self.fc1 = nn.Linear(26 * 26 * 32, 10)
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                return x

        # 로드는 항상 CPU로 강제 매핑 (CUDA/MPS 없는 환경에서도 100% 로드 가능)
        # 그 다음에 가능한 디바이스(cuda/mps/cpu)로 모델을 옮긴다.
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
        except Exception as load_exc:
            st.error(f"FashionMNIST 모델 로드 실패: {type(load_exc).__name__}: {load_exc}")
            return

        # 다양한 저장 형태 지원:
        # - torch.save(model.state_dict())
        # - torch.save({"state_dict": model.state_dict(), ...})
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # 실행 디바이스 선택: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model = SimpleCNN()
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # 샘플 데이터 준비 (FashionMNIST test 세트의 첫 8개)
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
        images, labels = [], []
        for idx in range(8):
            img, label = test_dataset[idx]
            images.append(img)
            labels.append(label)
        images_tensor = torch.stack(images).to(device)

        # 예측
        with torch.no_grad():
            outputs = model(images_tensor)
            preds = outputs.argmax(dim=1).cpu().numpy()

        # 클래스명 매핑
        classes = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ]
        st.success("FashionMNIST 모델 테스트 성공! 첫 8개 이미지 예측 결과:")

        import matplotlib.pyplot as plt
        import io

        set_korean_font_for_matplotlib()  # 한글 폰트 세팅

        # 2행 4열 그리드로 이미지와 예측/정답 출력
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i].squeeze(0).cpu().numpy(), cmap="gray")
            ax.set_title(f"정답: {classes[labels[i]]}\n예측: {classes[preds[i]]}")
            ax.axis("off")
        plt.tight_layout()

        # Streamlit에 출력
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        st.image(buf, caption="FashionMNIST 샘플 예측 결과", width="stretch")
    except Exception as exc:
        st.error(f"FashionMNIST 모델 테스트 실패: {type(exc).__name__}: {exc}")

def main():
    st.set_page_config(page_title="설치/모델 검증 대시보드", layout="wide")
    st.title("Python ML/AI 환경 설치 및 모델 아티팩트 검증")

    st.header("1. 설치된 패키지 버전 체크")
    check_library_versions()

    st.header("2. Scikit-learn 모델 아티팩트 테스트")
    test_sklearn_model()

    st.header("3. YOLO(v8, Ultralytics) 모델 아티팩트 테스트")
    test_yolo_model()

    st.header("4. PyTorch FashionMNIST 모델 아티팩트 테스트")
    test_fashionmnist_model()

    st.markdown("---")
    st.caption("Made for installation/test verification. 모델/노트북 파일이 없으면 먼저 학습/생성하세요.")

if __name__ == "__main__":
    main()
