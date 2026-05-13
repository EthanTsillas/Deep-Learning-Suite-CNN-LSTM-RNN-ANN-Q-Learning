from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.loaders import load_hr_tabular_loaders


class HRMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += y.numel()
    return correct / max(total, 1)


def train_ann(epochs: int = 12, lr: float = 1e-3, device: str = "cpu") -> dict:
    train_loader, test_loader, input_dim = load_hr_tabular_loaders()
    model = HRMLP(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_curve: list[float] = []

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        train_curve.append(evaluate(model, train_loader, device))

    return {
        "model": model,
        "train_accuracy": evaluate(model, train_loader, device),
        "test_accuracy": evaluate(model, test_loader, device),
        "train_curve": train_curve,
        "input_dim": input_dim,
    }
