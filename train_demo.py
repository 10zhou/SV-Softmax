import argparse
import os
import random
import sys
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import yaml
from tqdm import tqdm

import losses


# ---------------- Utils ----------------
def set_seed(seed=123):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(logits, targets):
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        return (pred == targets).float().mean().item()


def save_yaml(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


# ---------------- Data ----------------
def get_cifar10_loaders(data_root, batch_size, num_workers):
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(mean, std)])
    tf_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_set = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=tf_train)
    test_set = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=tf_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ---------------- Model ----------------
class ResNetBackbone(nn.Module):
    def __init__(self, arch="resnet18"):
        super().__init__()
        if arch == "resnet34":
            m = torchvision.models.resnet34(weights=None)
        else:
            m = torchvision.models.resnet18(weights=None)
        self.feature_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m

    def forward(self, x):
        return self.backbone(x)


class SVNet(nn.Module):
    requires_labels = True

    def __init__(self, num_classes=100, m=0.6):
        super().__init__()
        self.backbone = ResNetBackbone()
        self.head = losses.SVSoftmaxLoss(self.backbone.feature_dim, num_classes, m=m)

    def forward(self, x, labels=None):
        feats = self.backbone(x)              # (N, D)
        logits = self.head(feats, labels)     # (N, C)
        return logits


class VNet(nn.Module):
    requires_labels = True

    def __init__(self, num_classes=100, m=0.6):
        super().__init__()
        self.backbone = ResNetBackbone()
        self.head = losses.VirtualSoftmax(self.backbone.feature_dim, num_classes)

    def forward(self, x, labels=None):
        feats = self.backbone(x)              # (N, D)
        logits = self.head(feats, labels)     # (N, C)
        return logits


class NormLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight.data)

    def forward(self, input):
        input = F.normalize(input, dim=1)
        weight = F.normalize(self.weight, dim=1)
        return F.linear(input, weight)


def build_model(num_classes=100, cfg=None):
    if cfg.method == "svsoftmax":
        return SVNet(num_classes=num_classes, m=cfg.m)
    elif cfg.method == "virtual_softmax":
        return VNet(num_classes=num_classes)
    else:
        if cfg.arch == "resnet34":
            model = torchvision.models.resnet34(weights=None)
        else:
            model = torchvision.models.resnet18(weights=None)
        if cfg.method == "arcface" or cfg.method == "normface":
            model.fc = NormLinear(model.fc.in_features, num_classes)
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes, bias=False)
        return model


def build_criterion(name: str):
    name = name.lower()
    if name == "arcface":
        return losses.ArcFaceLoss()
    elif name == "normface":
        return losses.NormFaceLoss()
    elif name == "virtual_softmax":
        return losses.CrossEntropyLoss()
    elif name == "svsoftmax":
        return losses.CrossEntropyLoss()
    else:
        return losses.CrossEntropyLoss()


# --------------- Train/Eval ---------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    train_bar = tqdm(loader, file=sys.stdout)
    for _, (images, targets) in enumerate(train_bar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                if getattr(model, "requires_labels", False):
                    logits = model(images, labels=targets)
                else:
                    logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if getattr(model, "requires_labels", False):
                logits = model(images, labels=targets)
            else:
                logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        n += bs
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, targets) * bs
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    ce = criterion or nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
    return total_loss / n, total_acc / n


# ---------------- Config ----------------
@dataclass
class TrainConfig:
    arch: str = "resnet18"
    method: str = "ce"
    epochs: int = 100
    batch_size: int = 128
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    num_workers: int = 16
    seed: int = 123
    fp16: bool = False
    eval_interval: int = 10
    data_root: str = "./datasets"
    out_dir: str = "runs/benchmark"
    m: float = 0.6
    loop_num: int = 0

# --------------- Main ---------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--method", type=str, default="svsoftmax", choices=["svsoftmax", "virtual_softmax", "arcface",
                                                                            "normface", "ce"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--data-root", type=str, default="../datasets")
    parser.add_argument("--out-dir", type=str, default="runs/")
    parser.add_argument("--margin", type=float, default=0.6)
    parser.add_argument("--loop_num", type=int, default=0)
    args = parser.parse_args()

    cfg = TrainConfig(arch=args.arch, method=args.method, epochs=args.epochs,
                      batch_size=args.batch_size, lr=args.lr,
                      weight_decay=args.weight_decay, momentum=args.momentum,
                      num_workers=args.num_workers, seed=args.seed, fp16=bool(args.fp16),
                      eval_interval=args.eval_interval, data_root=args.data_root,
                      out_dir=args.out_dir, m=args.margin, loop_num=args.loop_num)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader = get_cifar10_loaders(cfg.data_root, cfg.batch_size, cfg.num_workers)
    model = build_model(num_classes=10, cfg=cfg).to(device)
    criterion = build_criterion(cfg.method if cfg.method != "ce" else "ce")
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.fp16 and torch.cuda.is_available()))

    # 训练
    best_acc, last_val = 0.0, {"loss": None, "acc": None}
    logs = []
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        if (epoch % cfg.eval_interval == 0) or (epoch == cfg.epochs):
            val_loss, val_acc = evaluate(model, test_loader, device, criterion)
            best_acc = max(best_acc, val_acc)
            last_val = {"loss": val_loss, "acc": val_acc}
            logs.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc,
                         "lr": train_scheduler.get_last_lr()[0]})
            print(f"[{cfg.method}] Epoch {epoch:03d} | "
                  f"train {train_loss:.4f}/{train_acc:.4f} | val {val_loss:.4f}/{val_acc:.4f}")
        train_scheduler.step()

    # Save YAML (hyperparameters + final and optimal + each validation log)
    yaml_path = os.path.join(cfg.out_dir, f"{cfg.method}_{cfg.loop_num}.yaml")
    payload = {
        "config": asdict(cfg),
        "env": {"device": str(device),
                "num_params": int(sum(p.numel() for p in model.parameters()))},
        "results": {
            "best_val_acc": float(best_acc),
            "final_val_loss": None if last_val["loss"] is None else float(last_val["loss"]),
            "final_val_acc": None if last_val["acc"] is None else float(last_val["acc"]),
            "val_logs": logs
        }
    }
    save_yaml(payload, yaml_path)
    print(f"[Saved YAML] {yaml_path}")


if __name__ == "__main__":
    main()
