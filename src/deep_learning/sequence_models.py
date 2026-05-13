from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.loaders import load_news_sequence_loaders


class NewsSequenceClassifier(nn.Module):
    def __init__(
        self,
        model_type: str = "lstm",
        vocab_size: int = 8000,
        embed_dim: int = 32,
        hidden_dim: int = 48,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if model_type not in {"rnn", "lstm"}:
            raise ValueError("model_type must be 'rnn' or 'lstm'")
        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        recurrent_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.recurrent = recurrent_cls(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, num_classes),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(sequence)
        lengths = (sequence != 0).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.recurrent(packed)
        if self.model_type == "lstm":
            hidden = hidden[0]
        return self.classifier(hidden[-1])


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


def train_sequence_model(
    model_type: str = "lstm",
    epochs: int = 6,
    lr: float = 0.003,
    device: str = "cpu",
    max_rows: int = 6000,
) -> dict:
    train_loader, test_loader, vocab_size, num_classes = load_news_sequence_loaders(max_rows=max_rows)
    model = NewsSequenceClassifier(model_type=model_type, vocab_size=vocab_size, num_classes=num_classes).to(device)
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
        "vocab_size": vocab_size,
    }
