"""
Execute `lecture_install_verify.ipynb` code cells in one shot,
then train and save a simple PyTorch FashionMNIST GPU model.

Usage (from project root):
  uv run python notebooks/lecture_install_verify_all.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def execute_notebook_cells(nb_path: Path) -> None:
    """Executes all code cells in a Jupyter notebook."""
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = [c for c in data.get("cells", []) if c.get("cell_type") == "code"]
    if not cells:
        print("No code cells found. Nothing to execute.")
        return

    print(f"Executing {len(cells)} code cells from: {nb_path}")

    ns: dict[str, object] = {"__name__": "__main__"}

    for idx, cell in enumerate(cells, start=1):
        code_lines = cell.get("source", [])
        code = "".join(code_lines) if isinstance(code_lines, list) else str(code_lines)
        print(f"\n--- Cell {idx}/{len(cells)} ---")
        try:
            compiled = compile(code, f"{nb_path.name}:cell{idx}", "exec")
            exec(compiled, ns, ns)
        except Exception as exc:
            print(f"Cell {idx} failed with {type(exc).__name__}: {exc}")
            traceback.print_exc()
            raise

    print("\nAll cells executed successfully.")


def train_and_save_fashionmnist_model():
    """
    Train a simple CNN on FashionMNIST with GPU (if available), MPS (Apple Silicon/M1/M2), or CPU and save the model.
    Model will be saved as models/fashionmnist_cnn.pt
    """
    print("\n==== PyTorch FashionMNIST Train & Save START ====\n")
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import datasets, transforms

        # Device selection with support for MPS (Apple M1/M2)
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        except Exception as dev_exc:
            print(f"Device selection failed, defaulting to CPU: {dev_exc}")
            device = torch.device("cpu")

        print(f"Using device: {device}")

        # Define a minimal simple CNN for FashionMNIST
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

        # Data
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # One epoch of quick training for smoke-test purposes
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Use try-except in case MPS can't handle some calls
            try:
                data, target = data.to(device), target.to(device)
            except Exception as to_exc:
                print(f"tensor.to(device) failed: {to_exc}, using CPU fallback")
                device = torch.device("cpu")
                data, target = data.to(device), target.to(device)
                model.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            # for install/demo: just a few batches
            if batch_idx >= 2:
                break

        # Save the model
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "fashionmnist_cnn.pt"
        try:
            torch.save(model.state_dict(), model_path)
            print(f"FashionMNIST CNN model saved to: {model_path.resolve()}")
        except Exception as save_exc:
            print(f"Model saving failed: {save_exc}")
            raise
        print("==== PyTorch FashionMNIST Train & Save DONE ====\n")
    except Exception as exc:
        print(f"FashionMNIST PyTorch model creation failed: {exc}")
        traceback.print_exc()
        raise

def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")

    nb_path = Path(__file__).resolve().parent / "lecture_install_verify.ipynb"
    if not nb_path.is_file():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    project_root = nb_path.parent.parent
    pyproject = project_root / "pyproject.toml"
    uv_lock = project_root / "uv.lock"
    if not pyproject.is_file() or not uv_lock.is_file():
        print(f"Project root detection failed.")
        print(f"  expected pyproject: {pyproject}")
        print(f"  expected uv.lock:   {uv_lock}")
        print()
        print("Run from the project root like:")
        print("  cd c:\\2026\\gpu_build")
        print("  uv run python notebooks/lecture_install_verify_all.py")
        raise SystemExit(2)

    os.chdir(project_root)

    try:
        import matplotlib  # noqa: F401
    except Exception as exc:
        print("Preflight import failed: matplotlib is not available.")
        print(f"  cwd: {Path.cwd()}")
        print(f"  python: {sys.executable}")
        print()
        print("Make sure you run with uv from the project root:")
        print("  cd c:\\2026\\gpu_build")
        print("  uv sync")
        print("  uv run python notebooks/lecture_install_verify_all.py")
        print()
        raise

    # 1. Execute all notebook code cells
    try:
        execute_notebook_cells(nb_path)
    except Exception as cell_exc:
        print(f"Notebook cell execution failed: {cell_exc}")
        print("Continuing to the next step...")

    # 2. PyTorch FashionMNIST GPU/MPS/CPU smoke model training and save
    try:
        train_and_save_fashionmnist_model()
    except Exception as model_exc:
        print(f"Model training/saving failed: {model_exc}")
        print("Done (with errors above).")


if __name__ == "__main__":
    try:
        main()
    except Exception as main_exc:
        print(f"Fatal error running script: {main_exc}")
        sys.exit(1)
