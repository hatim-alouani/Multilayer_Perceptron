#!/usr/bin/env python3
"""Split the raw dataset into a training set and a validation set."""

import argparse
import os

import data


def parse_args():
    p = argparse.ArgumentParser(description="split the WDBC dataset")
    p.add_argument("dataset", help="path to the raw csv (32 columns, no header)")
    p.add_argument("--train-size", type=float, default=0.8,
                   help="fraction kept for training (default 0.8)")
    p.add_argument("--seed", type=int, default=42,
                   help="rng seed, for a reproducible split")
    p.add_argument("--out-dir", default=".", help="where to write the two csv")
    return p.parse_args()


def main():
    args = parse_args()
    df = data.load_raw(args.dataset)
    _, y = data.features_labels(df)

    train_idx, valid_idx = data.stratified_indices(
        y, train_size=args.train_size, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "data_train.csv")
    valid_path = os.path.join(args.out_dir, "data_valid.csv")

    df.iloc[train_idx].to_csv(train_path, header=False, index=False)
    df.iloc[valid_idx].to_csv(valid_path, header=False, index=False)

    n_train_m = int((y[train_idx] == "M").sum())
    n_valid_m = int((y[valid_idx] == "M").sum())
    print(f"total          : {len(df)} samples")
    print(f"{train_path}: {len(train_idx)} samples "
          f"({n_train_m} M / {len(train_idx) - n_train_m} B)")
    print(f"{valid_path}: {len(valid_idx)} samples "
          f"({n_valid_m} M / {len(valid_idx) - n_valid_m} B)")


if __name__ == "__main__":
    main()
