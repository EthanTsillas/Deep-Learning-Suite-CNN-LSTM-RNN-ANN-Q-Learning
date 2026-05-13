from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATA_ARCHIVE = DATA_DIR / "datasets.zip"
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z']+")


def extract_data_if_needed() -> None:
    expected = ["hrdata3.csv", "news.csv"]
    if all((DATA_DIR / name).exists() for name in expected):
        return
    if DATA_ARCHIVE.exists():
        with zipfile.ZipFile(DATA_ARCHIVE, "r") as archive:
            archive.extractall(DATA_DIR)


def require_file(filename: str) -> Path:
    extract_data_if_needed()
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. See data/README.md for the expected data files.")
    return path


def read_delimited_file(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        header = file.readline()
    sep = "\t" if header.count("\t") > header.count(",") else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [str(col).strip().replace("\ufeff", "") for col in df.columns]
    return df


def require_columns(df: pd.DataFrame, filename: str, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{filename} is missing columns {missing}. Found columns: {list(df.columns)}")


def load_hr_tabular_loaders(
    batch_size: int = 128,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[DataLoader, DataLoader, int]:
    df = read_delimited_file(require_file("hrdata3.csv"))
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    require_columns(df, "hrdata3.csv", ["target"])

    y = df["target"].astype(int).to_numpy()
    x_df = df.drop(columns=["target", "enrollee_id"], errors="ignore")
    x_df = x_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x = StandardScaler().fit_transform(x_df.to_numpy(dtype=np.float32)).astype("float32")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long))
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=batch_size), x.shape[1]


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(str(text))]


def balanced_sample(df: pd.DataFrame, target_col: str, max_rows: int, random_state: int) -> pd.DataFrame:
    labels = sorted(df[target_col].dropna().unique().tolist())
    per_class = max(1, max_rows // max(1, len(labels)))
    parts = []
    for label in labels:
        group = df[df[target_col] == label]
        parts.append(group.sample(n=min(len(group), per_class), random_state=random_state))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def load_news_sequence_loaders(
    batch_size: int = 128,
    max_rows: int = 6000,
    max_vocab: int = 8000,
    max_len: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[DataLoader, DataLoader, int, int]:
    df = read_delimited_file(require_file("news.csv"))
    require_columns(df, "news.csv", ["title", "text", "target"])
    df = df[["title", "text", "target"]].dropna().reset_index(drop=True)
    df = balanced_sample(df, "target", max_rows=max_rows, random_state=random_state)

    labels = df["target"].astype(int).to_numpy()
    texts = (df["title"].astype(str) + " " + df["text"].astype(str)).tolist()
    tokenized = [tokenize(text)[:max_len] for text in texts]

    counter = Counter(token for doc in tokenized for token in doc)
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in counter.most_common(max_vocab - len(vocab)):
        vocab[token] = len(vocab)

    x = np.zeros((len(tokenized), max_len), dtype=np.int64)
    for i, doc in enumerate(tokenized):
        ids = [vocab.get(token, vocab["<unk>"]) for token in doc]
        x[i, : len(ids)] = ids

    x_train, x_test, y_train, y_test = train_test_split(
        x, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long))
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=batch_size), len(vocab), 2
