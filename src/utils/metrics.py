from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np


@dataclass
class ExperimentResult:
    name: str
    train_accuracy: float
    test_accuracy: float
    epochs: int
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] == 0:
        raise ValueError("Cannot compute accuracy for an empty target array")
    return float((y_true == y_pred).mean())


def save_metrics(results: list[ExperimentResult] | list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [r.to_dict() if hasattr(r, "to_dict") else r for r in results]
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
