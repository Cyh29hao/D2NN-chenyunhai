import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vit_b_16
from tqdm.auto import tqdm


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
        # 使用 raw 参数 + sigmoid 约束到 [0, 2pi]，更贴近仓库相位训练逻辑
        self.phase_raw = nn.Parameter(torch.zeros(channels, height, width))

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        phase = 2 * torch.pi * torch.sigmoid(self.phase_raw)
        return field * torch.exp(1j * phase)


class D2NN12WithSharedHead(nn.Module):
    """
    12-layer diffractive stack + same FC classifier design as ViT baseline.
    """

    def __init__(self, img_size: int = 224, channels: int = 3, num_classes: int = 10, fc_hidden_dim: int = 512):
        super().__init__()
        self.layers = nn.ModuleList(
            [DiffractiveLayer(channels, img_size, img_size) for _ in range(12)]
        )

        # 参考仓库中的角谱传播写法：使用固定传播核，而非恒等传播
        wl = 532e-9
        pixel_size = 8e-6
        distance = 0.01
        fx = np.fft.fftshift(np.fft.fftfreq(img_size, d=pixel_size))
        fy = np.fft.fftshift(np.fft.fftfreq(img_size, d=pixel_size))
        fxx, fyy = np.meshgrid(fx, fy)
        inside = (1.0 / wl) ** 2 - fxx ** 2 - fyy ** 2
        inside[inside < 0] = 0
        kz = 2 * np.pi * np.sqrt(inside)
        h = np.exp(1j * kz * distance).astype(np.complex64)
        self.register_buffer("transfer", torch.from_numpy(h))

        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc_head = SharedFCHead(channels * 8 * 8, hidden_dim=fc_hidden_dim, num_classes=num_classes)

    def propagate(self, field: torch.Tensor) -> torch.Tensor:
        spec = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
        out_spec = spec * self.transfer
        return torch.fft.ifft2(torch.fft.ifftshift(out_spec, dim=(-2, -1)))

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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0.0
    last_acc = 0.0
    last_loss = math.nan

    model.to(device)
    for epoch in range(epochs):
        model.train()
        running = 0.0
        samples = 0

        for x, y in tqdm(train_loader, desc=f"{model.__class__.__name__} epoch {epoch+1}/{epochs}"):
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


def select_phase_only_params(model: nn.Module):
    """
    阶段一：锁住大部分参数，只训练相位层 + 分类头。
    仅训练 phase 在 CIFAR10 上通常过难，加入 FC 头可避免准确率长期卡在随机水平。
    """
    for p in model.parameters():
        p.requires_grad = False
    params_to_update = []
    for name, p in model.named_parameters():
        if ("phase_raw" in name) or ("fc_head" in name):
            p.requires_grad = True
            params_to_update.append(p)
    return params_to_update


def unfreeze_all_params(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def train_d2nn_two_stage(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                         phase_epochs: int = 12, finetune_epochs: int = 40,
                         phase_lr: float = 2e-3, finetune_lr: float = 3e-4) -> RunMetrics:
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc = 0.0
    last_acc = 0.0
    last_loss = math.nan

    print("\n[Stage-1] 只训练相位层（锁定其余参数）")
    phase_params = select_phase_only_params(model)
    optimizer = torch.optim.AdamW(phase_params, lr=phase_lr, weight_decay=1e-4)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(phase_epochs, 1))
    for epoch in range(phase_epochs):
        model.train()
        running = 0.0
        samples = 0
        for x, y in tqdm(train_loader, desc=f"D2NN phase-only {epoch+1}/{phase_epochs}"):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            samples += y.size(0)
        scheduler1.step()
        last_loss = running / max(samples, 1)
        last_acc = evaluate(model, val_loader, device)
        best_acc = max(best_acc, last_acc)
        print(f"[Stage-1][epoch={epoch+1}] train_loss={last_loss:.4f} val_acc={last_acc*100:.2f}%")

    print("\n[Stage-2] 解锁全参数联合微调")
    all_params = unfreeze_all_params(model)
    optimizer = torch.optim.AdamW(all_params, lr=finetune_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(finetune_epochs, 1))
    for epoch in range(finetune_epochs):
        model.train()
        running = 0.0
        samples = 0
        for x, y in tqdm(train_loader, desc=f"D2NN finetune {epoch+1}/{finetune_epochs}"):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running += loss.item() * y.size(0)
            samples += y.size(0)
        scheduler.step()
        last_loss = running / max(samples, 1)
        last_acc = evaluate(model, val_loader, device)
        best_acc = max(best_acc, last_acc)
        print(f"[Stage-2][epoch={epoch+1}] train_loss={last_loss:.4f} val_acc={last_acc*100:.2f}%")

    return RunMetrics(
        model=model.__class__.__name__ + "_TwoStage",
        epochs=phase_epochs + finetune_epochs,
        best_val_acc=best_acc,
        last_val_acc=last_acc,
        train_loss=last_loss,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ViT-Base and D2NN-12 on CIFAR-10 using same FC head.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--d2nn-phase-epochs", type=int, default=12)
    parser.add_argument("--d2nn-finetune-epochs", type=int, default=40)
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
    d2nn_metrics = train_d2nn_two_stage(
        d2nn,
        train_loader,
        val_loader,
        device=device,
        phase_epochs=args.d2nn_phase_epochs,
        finetune_epochs=args.d2nn_finetune_epochs,
    )

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
