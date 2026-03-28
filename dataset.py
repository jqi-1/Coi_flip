"""
CoinFlipDataset — loads coin flip data from CSV (training) or Excel (test).

CSV format (produced by generate_data.py):
    flip_0, flip_1, ..., flip_{window-1}, label

Excel format (user-supplied test file):
    Any sheet with at least one column containing only 0s and 1s.
    The dataset builds sliding windows from that column automatically.
"""

import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CoinFlipDataset(Dataset):
    def __init__(self, path: str, window: int = 10):
        self.window = window
        if path.endswith(".xlsx") or path.endswith(".xls"):
            self.X, self.y = self._load_excel(path)
        else:
            self.X, self.y = self._load_csv(path)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_csv(self, path: str):
        df = pd.read_csv(path)
        feature_cols = [c for c in df.columns if c.startswith("flip_")]
        if feature_cols:
            label_col = "label" if "label" in df.columns else df.columns[-1]
        else:
            warnings.warn(
                f"{path}: expected columns named 'flip_N'; "
                "falling back to positional (first N cols = features, last = label)."
            )
            feature_cols = df.columns[: self.window].tolist()
            label_col = df.columns[-1]

        X = torch.FloatTensor(df[feature_cols].values)
        y = torch.FloatTensor(df[label_col].values)
        return X, y

    def _load_excel(self, path: str):
        df = pd.read_excel(path, engine="openpyxl")
        flips = self._detect_flip_column(df).values
        if len(flips) < self.window + 1:
            raise ValueError(
                f"Excel file has only {len(flips)} rows; need at least {self.window + 1} "
                f"to form one sample with window={self.window}."
            )
        windows = np.lib.stride_tricks.sliding_window_view(flips, self.window + 1)
        X = torch.FloatTensor(windows[:, :-1])
        y = torch.FloatTensor(windows[:, -1])
        return X, y

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_flip_column(self, df: pd.DataFrame) -> pd.Series:
        """Return the first column whose values are all 0 or 1."""
        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) > 0 and set(s.unique()).issubset({0, 1, 0.0, 1.0}):
                return s.astype(int).reset_index(drop=True)
        warnings.warn(
            "No binary (0/1) column found in Excel file; "
            "using first column and coercing to int."
        )
        return pd.to_numeric(df.iloc[:, 0], errors="coerce").fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
