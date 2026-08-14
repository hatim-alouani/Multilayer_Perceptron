#!/usr/bin/env python3
"""Load a trained model and evaluate it on a dataset."""

import argparse

import numpy as np

import data
import metrics
import network


def parse_args():
    p = argparse.ArgumentParser(description="predict with a trained MLP")
    p.add_argument("dataset", help="csv in the same raw format as the original")
    p.add_argument("--model", default="saved_model.npy")
    p.add_argument("--show", type=int, default=0,
                   help="print the first N per-sample predictions")
    return p.parse_args()


def main():
    args = parse_args()

    net = network.Network.load(args.model)

    df = data.load_raw(args.dataset)
    X_raw, y = data.features_labels(df)
    # the exact statistics from training, restored from the model file
    X = data.standardize(X_raw, net.mean, net.std)
    Y = data.one_hot(y)
    X, Y = data.to_network_layout(X, Y)

    A = net.forward(X)

    y_true = metrics.to_labels(Y)          # 0 = benign, 1 = malignant
    y_pred = metrics.to_labels(A)
    p_malignant = A[1, :]                  # row 1 = P(malignant)

    bce = network.binary_cross_entropy(y_true.astype(np.float64), p_malignant)

    print(f"samples            : {X.shape[1]}")
    print(f"binary cross-entropy: {bce:.4f}")
    print()
    print(metrics.report(y_true, y_pred))

    if args.show:
        print()
        print(f"{'#':>4}  {'true':>6}  {'pred':>6}  {'P(M)':>6}")
        for i in range(min(args.show, len(y_true))):
            mark = " " if y_true[i] == y_pred[i] else " <- wrong"
            print(f"{i:>4}  {data.CLASSES[y_true[i]]:>6}  "
                  f"{data.CLASSES[y_pred[i]]:>6}  {p_malignant[i]:>6.3f}{mark}")


if __name__ == "__main__":
    main()
