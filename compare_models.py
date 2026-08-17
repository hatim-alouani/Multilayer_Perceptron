import sys
import numpy as np
import matplotlib.pyplot as plt
import helpers

# bonus: display multiple learning curves on the same graph, to compare different models
CONFIGS = [
    {'label': 'lr=0.01, layers=24,24', 'learning_rate': 0.01, 'layer': [24, 24]},
    {'label': 'lr=0.05, layers=24,24', 'learning_rate': 0.05, 'layer': [24, 24]},
    {'label': 'lr=0.01, layers=16,16', 'learning_rate': 0.01, 'layer': [16, 16]},
]

EPOCHS = 100
BATCH_SIZE = 8

def train_and_get_val_loss_history(x_train, y_train, x_valid, y_valid, layer, learning_rate):
    sizes = [x_train.shape[1]] + layer + [2]
    rng = np.random.default_rng(0)

    helpers.weights = []
    helpers.biases = []
    for i in range(len(sizes) - 1):
        limit = np.sqrt(6 / sizes[i])
        w = rng.uniform(-limit, limit, (sizes[i], sizes[i + 1]))
        b = np.zeros((1, sizes[i + 1]))
        helpers.weights.append(w)
        helpers.biases.append(b)

    val_loss_history = []
    for epoch in range(EPOCHS):
        for start in range(0, len(x_train), BATCH_SIZE):
            x_batch = x_train[start:start + BATCH_SIZE]
            y_batch = y_train[start:start + BATCH_SIZE]
            prediction, l_list = helpers.forward(x_batch)
            dW, db = helpers.backward(x_batch, y_batch, l_list)
            helpers.weights, helpers.biases = helpers.gradient_descent(helpers.weights, helpers.biases, dW, db, learning_rate)

        val_out, val_list = helpers.forward(x_valid)
        val_loss_history.append(helpers.categorical_cross_entropy(y_valid, val_out))

    return val_loss_history

def main():
    x_train, y_train = helpers.load('data_train.csv')
    x_valid, y_valid = helpers.load('data_valid.csv')

    x_min = x_train.min(axis=0)
    x_max = x_train.max(axis=0)
    x_train = (x_train - x_min) / (x_max - x_min)
    x_valid = (x_valid - x_min) / (x_max - x_min)

    y_train = np.array([[1, 0] if v == 'B' else [0, 1] for v in y_train])
    y_valid = np.array([[1, 0] if v == 'B' else [0, 1] for v in y_valid])

    plt.figure()
    for cfg in CONFIGS:
        print("--- training '%s' ---" % cfg['label'])
        val_loss_history = train_and_get_val_loss_history(
            x_train, y_train, x_valid, y_valid, cfg['layer'], cfg['learning_rate'])
        plt.plot(val_loss_history, label=cfg['label'])

    plt.xlabel('epochs')
    plt.ylabel('validation loss')
    plt.title('Learning Curves — Model Comparison')
    plt.legend()
    plt.savefig('compare_curves.png')
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Error:', e)
        sys.exit(1)
