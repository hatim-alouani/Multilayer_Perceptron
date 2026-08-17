import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import helpers

def parse_args():
    parser = argparse.ArgumentParser(description='Train a multilayer perceptron.')
    parser.add_argument('--dataset', type=str, default='data_train.csv',
                         help='path to the training dataset')
    parser.add_argument('--val_dataset', type=str, default='data_valid.csv',
                         help='path to the validation dataset')
    parser.add_argument('--layer', type=int, nargs='+', default=[24, 24],
                         help='sizes of the hidden layers (at least 2 required, default [24, 24])')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--loss', type=str, default='categoricalCrossentropy',
                         choices=['categoricalCrossentropy'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    return parser.parse_args()

def main():
    args = parse_args()

    # load data
    x_train, y_train = helpers.load(args.dataset)
    x_valid, y_valid = helpers.load(args.val_dataset)

    print('x_train shape :', x_train.shape)
    print('x_valid shape :', x_valid.shape)

    # preprocessing
    x_min = x_train.min(axis=0)
    x_max = x_train.max(axis=0)
    x_train = (x_train - x_min) / (x_max - x_min)
    x_valid = (x_valid - x_min) / (x_max - x_min)

    y_train = np.array([[1, 0] if v == 'B' else [0, 1] for v in y_train])
    y_valid = np.array([[1, 0] if v == 'B' else [0, 1] for v in y_valid])

    # init weights: input size from the data, output fixed at 2 classes,
    # hidden layers configurable via --layer (at least 2 required, default [24, 24])
    hidden_layers = args.layer
    if len(hidden_layers) < 2:
        print('Warning: at least 2 hidden layers are required, using default [24, 24]')
        hidden_layers = [24, 24]

    sizes = [x_train.shape[1]] + hidden_layers + [2]
    rng = np.random.default_rng(0)

    for i in range(len(sizes) - 1):
        limit = np.sqrt(6 / sizes[i])
        w = rng.uniform(-limit, limit, (sizes[i], sizes[i + 1]))
        b = np.zeros((1, sizes[i + 1]))
        helpers.weights.append(w)
        helpers.biases.append(b)

    # train loop
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size

    loss_history = []
    val_loss_history = []
    acc_history = []
    val_acc_history = []

    # bonus: early stopping
    patience = 10
    best_val_loss = np.inf
    epochs_without_improvement = 0

    for epoch in range(epochs):
        for start in range(0, len(x_train), batch_size):
            x_batch = x_train[start:start + batch_size]
            y_batch = y_train[start:start + batch_size]
            prediction, l_list = helpers.forward(x_batch)
            dW, db = helpers.backward(x_batch, y_batch, l_list)
            helpers.weights, helpers.biases = helpers.gradient_descent(helpers.weights, helpers.biases, dW, db, learning_rate)

        out, out_list = helpers.forward(x_train)
        val_out, val_list = helpers.forward(x_valid)

        train_loss = helpers.categorical_cross_entropy(y_train, out)
        val_loss = helpers.categorical_cross_entropy(y_valid, val_out)
        train_acc = helpers.accuracy(y_train, out)
        val_acc = helpers.accuracy(y_valid, val_out)

        loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print('epoch %03d - loss: %.4f - val_loss: %.4f' % (epoch + 1, train_loss, val_loss))

        # bonus: early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print('early stopping: no improvement in val_loss for %d epochs' % patience)
            break

    # save model
    np.save('saved_model.npy', {
        'sizes': sizes,
        'weights': helpers.weights,
        'biases': helpers.biases,
        'x_min': x_min,
        'x_max': x_max,
    }, allow_pickle=True)
    print("> saving model './saved_model.npy' to disk...")

    # save training history
    history = {
        'loss': [float(v) for v in loss_history],
        'val_loss': [float(v) for v in val_loss_history],
        'accuracy': [float(v) for v in acc_history],
        'val_accuracy': [float(v) for v in val_acc_history],
    }
    with open('history.json', 'w') as f:
        json.dump(history, f)
    print("> saving history './history.json' to disk...")

    # learning curves (mandatory: two graphs displayed at the end of training)
    plt.figure()
    plt.plot(history['loss'], label='training loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.savefig('loss_curve.png')

    plt.figure()
    plt.plot(history['accuracy'], label='training acc')
    plt.plot(history['val_accuracy'], label='validation acc')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('Learning Curves')
    plt.legend()
    plt.savefig('accuracy_curve.png')

    plt.show()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Error:', e)
        sys.exit(1)
