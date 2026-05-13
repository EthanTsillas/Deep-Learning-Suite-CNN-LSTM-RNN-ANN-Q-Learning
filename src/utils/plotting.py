from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_curve(values: list[float], title: str, ylabel: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(values) + 1), values, marker="o")
    plt.title(title)
    plt.xlabel("Episode" if "Reward" in ylabel else "Epoch")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def moving_average(values: list[float], window: int = 10) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    return [float(arr[max(0, i - window + 1): i + 1].mean()) for i in range(len(arr))]


def plot_q_learning_moving_average(rewards: list[float], output_path: str | Path, window: int = 10) -> list[float]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    smoothed = moving_average(rewards, window=window)
    episodes = range(1, len(rewards) + 1)
    plt.figure()
    plt.plot(episodes, rewards, alpha=0.35, label="Raw episode reward")
    plt.plot(episodes, smoothed, linewidth=2, label=f"{window}-episode moving average")
    plt.title("Q-Learning Reward Moving Average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return smoothed
