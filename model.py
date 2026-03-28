"""
Neural network models for coin flip prediction.

Both models return a raw logit (no sigmoid applied), so they pair
directly with torch.nn.BCEWithLogitsLoss.

MLPModel  — simple feed-forward network (default)
LSTMModel — recurrent network that treats each flip as a timestep
"""

import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """
    Multi-layer perceptron that takes a flat window of flips as input.

    Architecture: Linear(window -> 64) -> ReLU -> Dropout(0.3) -> Linear(64 -> 1)
    """

    def __init__(self, window: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window)  →  logit: (batch,)
        return self.net(x).squeeze(-1)


class LSTMModel(nn.Module):
    """
    LSTM that processes each flip in the window as a single-feature timestep.

    Architecture: LSTM(input=1, hidden=32) -> Linear(32 -> 1) on last hidden state
    """

    def __init__(self, window: int = 10, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window)  →  unsqueeze to (batch, window, 1)
        x = x.unsqueeze(-1)
        _, (h, _) = self.lstm(x)   # h: (1, batch, hidden)
        return self.head(h.squeeze(0)).squeeze(-1)  # (batch,)


def build_model(model_type: str, window: int) -> nn.Module:
    """Factory function used by train.py and evaluate.py."""
    if model_type == "lstm":
        return LSTMModel(window=window)
    return MLPModel(window=window)
