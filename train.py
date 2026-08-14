#!/usr/bin/env python3
"""Train the multilayer perceptron and save the model."""

import argparse
import json

import numpy as np

import data
import metrics
import network
import plots


def parse_args():
    p = argparse.ArgumentParser(description="train the MLP")
    p.add_argument("--train-set", default="data_train.csv")
    p.add_argument("--valid-set", default="data_valid.csv")
    p.add_argument("--layer", type=int, nargs="+", default=[24, 24],
                   help="hidden layer sizes (at least two)")
    p.add_argument("--epochs", type=int, default=84)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=0.0314)
    p.add_argument("--loss", default="categoricalCrossentropy",
                   choices=sorted(network.LOSSES))
    p.add_argument("--activation", default="sigmoid",
                   choices=["sigmoid", "relu", "tanh"])
    p.add_argument("--weights-initializer", default="heUniform",
                   choices=sorted(network.INITIALIZERS))
    p.add_argument("--optimizer", default="sgd",
                   choices=["sgd", "momentum", "adam"])
    p.add_argument("--early-stopping", type=int, default=0,
                   help="patience in epochs; 0 disables it")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-out", default="saved_model.npy")
    p.add_argument("--history-out", default="history.json")
    p.add_argument("--gradient-check", action="store_true",
                   help="verify backprop against finite differences, then exit")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def load_split(path, mean=None, std=None):
    df = data.load_raw(path)
    X, y = data.features_labels(df)
    if mean is None:
        mean, std = data.fit_standardizer(X)
    X = data.standardize(X, mean, std)
    Y = data.one_hot(y)
    return data.to_network_layout(X, Y) + (mean, std)


def main():
    args = parse_args()

    if len(args.layer) < 2:
        raise SystemExit("--layer needs at least two hidden layers")

    # statistics come from the training set only, then are reused verbatim
    X_train, Y_train, mean, std = load_split(args.train_set)
    X_valid, Y_valid, _, _ = load_split(args.valid_set, mean, std)

    print(f"x_train shape : {X_train.T.shape}")
    print(f"x_valid shape : {X_valid.T.shape}")

    layer_sizes = [X_train.shape[0]] + list(args.layer) + [Y_train.shape[0]]
    net = network.Network(
        layer_sizes,
        activation=args.activation,
        output_activation="softmax",
        weights_initializer=args.weights_initializer,
        optimizer=args.optimizer,
        seed=args.seed,
    )
    net.mean = mean
    net.std = std
    print("topology      : " + " -> ".join(str(n) for n in layer_sizes))

    if args.gradient_check:
        network.gradient_check(net, X_train[:, :8], Y_train[:, :8])
        return

    loss_fn = network.LOSSES[args.loss]
    rng = np.random.default_rng(args.seed)
    m = X_train.shape[1]

    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
    best_val = np.inf
    best_state = None
    stale = 0

    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(m)
        X_shuffled = X_train[:, perm]
        Y_shuffled = Y_train[:, perm]

        for start in range(0, m, args.batch_size):
            X_batch = X_shuffled[:, start:start + args.batch_size]
            Y_batch = Y_shuffled[:, start:start + args.batch_size]
            net.forward(X_batch)
            net.backward(Y_batch)
            net.update(args.learning_rate)

        A_train = net.forward(X_train)
        A_valid = net.forward(X_valid)
        loss = loss_fn(Y_train, A_train)
        val_loss = loss_fn(Y_valid, A_valid)
        acc = metrics.accuracy(Y_train, A_train)
        val_acc = metrics.accuracy(Y_valid, A_valid)

        history["loss"].append(loss)
        history["val_loss"].append(val_loss)
        history["acc"].append(acc)
        history["val_acc"].append(val_acc)

        print(f"epoch {epoch:02d}/{args.epochs} - loss: {loss:.4f} - "
              f"val_loss: {val_loss:.4f} - acc: {acc:.4f} - "
              f"val_acc: {val_acc:.4f}")

        if args.early_stopping:
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = ([l.W.copy() for l in net.layers],
                              [l.b.copy() for l in net.layers])
                stale = 0
            else:
                stale += 1
                if stale >= args.early_stopping:
                    print(f"> early stopping at epoch {epoch} "
                          f"(no improvement for {stale} epochs)")
                    break

    if args.early_stopping and best_state is not None:
        for layer, W, b in zip(net.layers, *best_state):
            layer.W, layer.b = W, b
        print(f"> restored best weights (val_loss {best_val:.4f})")

    A_valid = net.forward(X_valid)
    print()
    print(metrics.report(metrics.to_labels(Y_valid), metrics.to_labels(A_valid)))
    print()

    print(f"> saving model '{args.model_out}' to disk...")
    net.save(args.model_out)
    with open(args.history_out, "w") as fh:
        json.dump(history, fh)

    if not args.no_plot:
        plots.plot_history(history, show=False)


if __name__ == "__main__":
    main()
