from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    Dataset = object
    DataLoader = None


class CsvImageDataset(Dataset):
    def __init__(self, csv_path: Path, image_size: int = 224, is_train: bool = False) -> None:
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = cv2.imread(str(row["image_path"]))
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        if self.is_train and np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
        x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y


def build_model(model_name: str = "mobilenet_v3_small", num_classes: int = 2):
    if torch is None:
        raise RuntimeError("PyTorch is not available. Install requirements first.")
    try:
        import timm

        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        return model
    except Exception:
        from torchvision.models import mobilenet_v3_small

        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model


def train_torch_model(
    train_csv: Path,
    val_csv: Path,
    model_path: Path,
    model_name: str = "mobilenet_v3_small",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> dict[str, float]:
    if torch is None:
        raise RuntimeError("PyTorch is not available. Install requirements first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = CsvImageDataset(train_csv, is_train=True)
    val_ds = CsvImageDataset(val_csv, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(model_name=model_name, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
        val_acc = correct / max(total, 1)
        if val_acc > best_val:
            best_val = val_acc
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_name": model_name, "state_dict": model.state_dict()}, model_path)

    return {"val_accuracy": float(best_val)}


def predict_torch_positive_score(model_path: Path, image_bgr: np.ndarray) -> float:
    if torch is None:
        raise RuntimeError("PyTorch is not available. Install requirements first.")

    ckpt = torch.load(model_path, map_location="cpu")
    model_name = ckpt.get("model_name", "mobilenet_v3_small")
    model = build_model(model_name=model_name, num_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    x = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
    return float(probs[1])
