from __future__ import annotations

import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LeNetStyleCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def make_loaders(batch_size: int = 128, test_size: float = 0.2, random_state: int = 42) -> tuple[DataLoader, DataLoader]:
    digits = load_digits()
    x = digits.images.astype("float32") / 16.0
    y = digits.target.astype("int64")
    x = x[:, None, :, :]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=batch_size)


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


def train_cnn(epochs: int = 16, lr: float = 0.003, device: str = "cpu") -> dict:
    train_loader, test_loader = make_loaders()
    model = LeNetStyleCNN().to(device)
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
    }
