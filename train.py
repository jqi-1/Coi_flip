"""
Train a neural network on coin flip sequences.

Usage:
    python train.py
    python train.py --model lstm --epochs 30 --batch_size 512
    python train.py --train_csv data/train.csv --window 10 --lr 1e-3

The checkpoint saved at the end includes model weights, window size, and model
type so that evaluate.py can reconstruct the model without extra flags.
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CoinFlipDataset
from model import build_model


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    dataset = CoinFlipDataset(args.train_csv, window=args.window)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Training samples: {len(dataset)}")

    # --- Model ---
    model = build_model(args.model, args.window).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Training loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  acc={accuracy:.4f}")

    # --- Save checkpoint ---
    torch.save(
        {
            "model_state": model.state_dict(),
            "window": args.window,
            "model_type": args.model,
        },
        args.checkpoint,
    )
    print(f"\nCheckpoint saved to {args.checkpoint}")
    print("Note: ~0.50 accuracy is expected for a fair coin — this is a pipeline demo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train coin flip neural network")
    parser.add_argument("--train_csv", default="data/train.csv",
                        help="Path to training CSV (default: data/train.csv)")
    parser.add_argument("--window", type=int, default=10,
                        help="Sequence window size (default: 10)")
    parser.add_argument("--model", choices=["mlp", "lstm"], default="mlp",
                        help="Model architecture (default: mlp)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Mini-batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--checkpoint", default="checkpoint.pt",
                        help="Path to save model checkpoint (default: checkpoint.pt)")
    args = parser.parse_args()

    train(args)
