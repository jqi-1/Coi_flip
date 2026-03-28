"""
Generate a training dataset of fair coin flips.

Produces overlapping windows of `window` flips and the next flip as the label.
Output is saved as a CSV with columns flip_0..flip_{window-1} and label.

Usage:
    python generate_data.py
    python generate_data.py --n_flips 100000 --window 10 --seed 0 --out data/train.csv
"""

import argparse
import os

import numpy as np
import pandas as pd


def generate(n_flips: int, window: int, seed: int, out_path: str) -> None:
    rng = np.random.default_rng(seed)
    flips = rng.integers(0, 2, size=n_flips)  # 0 = tails, 1 = heads

    # Build overlapping windows: each row is `window` flips + 1 label
    windows = np.lib.stride_tricks.sliding_window_view(flips, window + 1)

    feature_cols = {f"flip_{i}": windows[:, i] for i in range(window)}
    df = pd.DataFrame(feature_cols)
    df["label"] = windows[:, -1]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} samples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate coin flip training data")
    parser.add_argument("--n_flips", type=int, default=50_000,
                        help="Total number of coin flips to generate (default: 50000)")
    parser.add_argument("--window", type=int, default=10,
                        help="Number of past flips used as features (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--out", default="data/train.csv",
                        help="Output CSV path (default: data/train.csv)")
    args = parser.parse_args()

    generate(args.n_flips, args.window, args.seed, args.out)
