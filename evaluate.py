"""
Evaluate a trained coin flip model on the test dataset.

Usage:
    python evaluate.py --test_xlsx data/test.xlsx
    python evaluate.py --test_xlsx data/test.xlsx --checkpoint checkpoint.pt

The checkpoint must have been created by train.py. It stores the model type
and window size, so no extra flags are needed to reconstruct the model.
"""

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import CoinFlipDataset
from model import build_model


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location=device)
    window = ckpt["window"]
    model_type = ckpt["model_type"]

    model = build_model(model_type, window).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {model_type.upper()} model (window={window}) from {args.checkpoint}")

    # --- Load test data ---
    dataset = CoinFlipDataset(args.test_xlsx, window=window)
    loader = DataLoader(dataset, batch_size=256)
    print(f"Test samples: {len(dataset)}")

    # --- Evaluation ---
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nTest accuracy: {accuracy:.4f}  ({correct}/{total} correct)")
    print("Note: ~0.50 accuracy is expected for a fair coin — this is a pipeline demo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate coin flip neural network")
    parser.add_argument("--test_xlsx", required=True,
                        help="Path to test Excel file (.xlsx)")
    parser.add_argument("--checkpoint", default="checkpoint.pt",
                        help="Path to model checkpoint (default: checkpoint.pt)")
    args = parser.parse_args()

    evaluate(args)
