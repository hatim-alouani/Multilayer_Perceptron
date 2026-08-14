"""Learning curves. Allowed by the subject: you may use a plotting library."""

import matplotlib
matplotlib.use("Agg")  # write to file, no display needed
import matplotlib.pyplot as plt


def plot_history(history, out_path="learning_curves.png", show=True):
    """Two side-by-side plots: loss and accuracy, training vs validation.

    history is a dict of lists: loss, val_loss, acc, val_acc.
    """
    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, history["loss"], label="training")
    ax_loss.plot(epochs, history["val_loss"], label="validation")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["acc"], label="training")
    ax_acc.plot(epochs, history["val_acc"], label="validation")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"> learning curves written to '{out_path}'")
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


def plot_comparison(histories, key="val_loss", out_path="comparison.png"):
    """Bonus: several runs on one graph. histories = {label: history_dict}."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, history in histories.items():
        ax.plot(range(1, len(history[key]) + 1), history[key], label=label)
    ax.set_title(key)
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
