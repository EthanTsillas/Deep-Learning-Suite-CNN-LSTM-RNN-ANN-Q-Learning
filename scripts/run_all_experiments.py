from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from sklearn.datasets import load_digits

from src.deep_learning.ann import train_ann
from src.deep_learning.cnn import train_cnn
from src.deep_learning.numpy_ops import laplacian_edge_detect, max_pool2d
from src.deep_learning.sequence_models import train_sequence_model
from src.rl.q_learning import train_q_learning
from src.utils.metrics import ExperimentResult, save_metrics
from src.utils.plotting import plot_curve, plot_q_learning_moving_average
from src.utils.seed import set_seed

RESULTS_DIR = ROOT / "results"

torch.set_num_threads(1)


def run_numpy_primitives() -> dict:
    digits = load_digits()
    image = digits.images[0].astype("float32") / 16.0
    edges = laplacian_edge_detect(image)
    pooled = max_pool2d(image)
    return {
        "name": "NumPy CNN primitives",
        "edge_map_shape": list(edges.shape),
        "pooled_shape": list(pooled.shape),
        "notes": "Ran Laplacian edge detection and max pooling on a scikit-learn digit image.",
    }


def main() -> None:
    set_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results: list[ExperimentResult | dict] = []

    ann = train_ann(epochs=12, lr=0.001, device=device)
    results.append(
        ExperimentResult(
            "ANN / MLP HR tabular classifier",
            ann["train_accuracy"],
            ann["test_accuracy"],
            12,
            notes=f"Trained on hrdata3.csv with {ann['input_dim']} standardized input features.",
        )
    )
    plot_curve(ann["train_curve"], "ANN Training Accuracy", "Accuracy", RESULTS_DIR / "ann_training_accuracy.png")

    cnn = train_cnn(epochs=16, lr=0.003, device=device)
    results.append(
        ExperimentResult(
            "CNN digits classifier",
            cnn["train_accuracy"],
            cnn["test_accuracy"],
            16,
            notes="Trained on the scikit-learn digits image dataset.",
        )
    )
    plot_curve(cnn["train_curve"], "CNN Training Accuracy", "Accuracy", RESULTS_DIR / "cnn_training_accuracy.png")

    rnn = train_sequence_model(model_type="rnn", epochs=4, lr=0.003, device=device, max_rows=2000)
    results.append(
        ExperimentResult(
            "RNN news text classifier",
            rnn["train_accuracy"],
            rnn["test_accuracy"],
            4,
            notes=f"Trained on a balanced news.csv subset with vocabulary size {rnn['vocab_size']}.",
        )
    )
    plot_curve(rnn["train_curve"], "RNN Training Accuracy", "Accuracy", RESULTS_DIR / "rnn_training_accuracy.png")

    lstm = train_sequence_model(model_type="lstm", epochs=4, lr=0.003, device=device, max_rows=2000)
    results.append(
        ExperimentResult(
            "LSTM news text classifier",
            lstm["train_accuracy"],
            lstm["test_accuracy"],
            4,
            notes=f"Trained on a balanced news.csv subset with vocabulary size {lstm['vocab_size']}.",
        )
    )
    plot_curve(lstm["train_curve"], "LSTM Training Accuracy", "Accuracy", RESULTS_DIR / "lstm_training_accuracy.png")

    q = train_q_learning(episodes=150)
    q_moving_average = plot_q_learning_moving_average(q["rewards"], RESULTS_DIR / "q_learning_moving_average.png", window=10)
    results.append(
        {
            "name": "Q-learning GridWorld",
            "average_last_50_reward": q["average_last_50_reward"],
            "final_10_episode_moving_average_reward": q_moving_average[-1],
            "moving_average_window": 10,
            "episodes": 150,
            "policy": q["policy"].tolist(),
            "notes": "Tabular Q-learning with epsilon-greedy exploration and Bellman updates.",
        }
    )
    plot_curve(q["rewards"], "Q-Learning Episode Rewards", "Reward", RESULTS_DIR / "q_learning_rewards.png")

    results.append(run_numpy_primitives())
    save_metrics(results, RESULTS_DIR / "metrics.json")
    print(f"Saved metrics and plots to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
