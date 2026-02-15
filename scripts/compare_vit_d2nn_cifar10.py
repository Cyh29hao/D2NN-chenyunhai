import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vit_b_16


@dataclass
class RunMetrics:
    model: str
    epochs: int
    best_val_acc: float
    last_val_acc: float
    train_loss: float


class SharedFCHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ViTWithSharedHead(nn.Module):
    def __init__(self, num_classes: int = 10, fc_hidden_dim: int = 512):
        super().__init__()
        self.backbone = vit_b_16(weights=None)
        in_dim = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Identity()
        self.fc_head = SharedFCHead(in_dim, hidden_dim=fc_hidden_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.fc_head(features)


class DiffractiveLayer(nn.Module):
    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        self.phase = nn.Parameter(torch.zeros(channels, height, width))

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return field * torch.exp(1j * self.phase)


class D2NN12WithSharedHead(nn.Module):
    """
    12-layer diffractive stack + same FC classifier design as ViT baseline.
    """

    def __init__(self, img_size: int = 32, channels: int = 3, num_classes: int = 10, fc_hidden_dim: int = 512):
        super().__init__()
        self.layers = nn.ModuleList(
            [DiffractiveLayer(channels, img_size, img_size) for _ in range(12)]
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc_head = SharedFCHead(channels * 8 * 8, hidden_dim=fc_hidden_dim, num_classes=num_classes)

    @staticmethod
    def propagate(field: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft2(torch.fft.fft2(field))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        field = x.to(torch.complex64)
        for layer in self.layers:
            field = layer(field)
            field = self.propagate(field)
        intensity = torch.abs(field) ** 2
        pooled = self.pool(intensity).flatten(1)
        return self.fc_head(pooled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(data_dir: Path, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    train_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_ds = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=train_t)
    test_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int, lr: float) -> RunMetrics:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0.0
    last_acc = 0.0
    last_loss = math.nan

    model.to(device)
    for epoch in range(epochs):
        model.train()
        running = 0.0
        samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running += loss.item() * y.size(0)
            samples += y.size(0)

        last_loss = running / max(samples, 1)
        last_acc = evaluate(model, val_loader, device)
        best_acc = max(best_acc, last_acc)
        print(f"epoch={epoch + 1:02d} train_loss={last_loss:.4f} val_acc={last_acc * 100:.2f}%")

    return RunMetrics(
        model=model.__class__.__name__,
        epochs=epochs,
        best_val_acc=best_acc,
        last_val_acc=last_acc,
        train_loss=last_loss,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ViT-Base and D2NN-12 on CIFAR-10 using same FC head.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/cifar10_vit_vs_d2nn.json"))
    # In notebook environments (ipykernel), extra args like
    # "-f /path/to/kernel.json" are injected into sys.argv.
    # parse_known_args keeps CLI behavior while safely ignoring them.
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[info] Ignoring unrecognized args: {unknown}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    set_seed(args.seed)
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(args.data_dir, args.batch_size, args.num_workers)

    print("\n=== Train ViT-Base + Shared FC ===")
    vit = ViTWithSharedHead(num_classes=10, fc_hidden_dim=512)
    vit_metrics = train_model(vit, train_loader, val_loader, device=device, epochs=args.epochs, lr=args.lr)

    print("\n=== Train D2NN-12 + Shared FC ===")
    d2nn = D2NN12WithSharedHead(img_size=224, channels=3, num_classes=10, fc_hidden_dim=512)
    d2nn_metrics = train_model(d2nn, train_loader, val_loader, device=device, epochs=args.epochs, lr=args.lr)

    payload: Dict[str, Dict] = {
        "vit_base_shared_fc": asdict(vit_metrics),
        "d2nn_12_shared_fc": asdict(d2nn_metrics),
    }
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(payload, indent=2))
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
