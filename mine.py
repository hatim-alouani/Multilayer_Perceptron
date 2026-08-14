import pandas as pd
import numpy as np

# ---------------------------------------------------------------- split
df = pd.read_csv('data.csv', header=None).sample(frac=1, random_state=0)
cut = int(len(df) * 0.8)
df.iloc[:cut].to_csv('data_train.csv', header=False, index=False)
df.iloc[cut:].to_csv('data_valid.csv', header=False, index=False)

# ---------------------------------------------------------------- load
def load(path):
    df = pd.read_csv(path, header=None)
    x = df.iloc[:, 2:].values.astype(float)
    y = df.iloc[:, 1].values
    return x, y

x_train, y_train = load('data_train.csv')
x_valid, y_valid = load('data_valid.csv')

# ---------------------------------------------------------- preprocess
x_min = x_train.min(axis=0)
x_max = x_train.max(axis=0)
x_train = (x_train - x_min) / (x_max - x_min)
x_valid = (x_valid - x_min) / (x_max - x_min)

y_train = np.array([[1, 0] if v == 'B' else [0, 1] for v in y_train])
y_valid = np.array([[1, 0] if v == 'B' else [0, 1] for v in y_valid])

# --------------------------------------------------------- init weights
sizes = [30, 24, 24, 2] #30 input features -> hidden layer of 24 neurons -> hidden layer of 24 neurons -> output layer of 2 neurons
weights = []
biases = []
rng = np.random.default_rng(0)

for i in range(len(sizes) - 1):
    limit = np.sqrt(6 / sizes[i])
    w = rng.uniform(-limit, limit, (sizes[i], sizes[i + 1]))
    b = np.zeros((1, sizes[i + 1]))
    weights.append(w)
    biases.append(b)

# ------------------------------------------------------- activations
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# ------------------------------------------------------------ forward
def forward(x):
    a = x
    a_list = [x]
    for i in range(len(weights)):
        z = a @ weights[i] + biases[i]
        if i == len(weights) - 1:
            a = softmax(z)
        else:
            a = sigmoid(z)
        a_list.append(a)
    return a, a_list

# ------------------------------------------------------------- loss
def loss(y, a):
    a = np.clip(a, 1e-15, 1 - 1e-15)
    return -np.sum(y * np.log(a)) / len(y)

# ---------------------------------------------------------- backward
def backward(x, y, a_list):
    m = len(x)
    dW = []
    db = []

    delta = a_list[-1] - y

    for i in range(len(weights) - 1, -1, -1):
        dw = a_list[i].T @ delta / m
        dbias = np.sum(delta, axis=0, keepdims=True) / m
        dW.insert(0, dw)
        db.insert(0, dbias)
        if i > 0:
            delta = (delta @ weights[i].T) * a_list[i] * (1 - a_list[i])

    return dW, db

# ----------------------------------------------------------- update
def update(weights, biases, dW, db, learning_rate):
    for i in range(len(weights)):
        weights[i] = weights[i] - learning_rate * dW[i]
        biases[i]  = biases[i]  - learning_rate * db[i]
    return weights, biases

# -------------------------------------------------------- train loop
learning_rate = 0.01
epochs = 100
batch_size = 8

for epoch in range(epochs):
    idx = rng.permutation(len(x_train))
    x_shuffled = x_train[idx]
    y_shuffled = y_train[idx]

    for start in range(0, len(x_train), batch_size):
        x_batch = x_shuffled[start:start + batch_size]
        y_batch = y_shuffled[start:start + batch_size]
        _, a_list = forward(x_batch)
        dW, db = backward(x_batch, y_batch, a_list)
        weights, biases = update(weights, biases, dW, db, learning_rate)

    out, _ = forward(x_train)
    val_out, _ = forward(x_valid)
    print('epoch %03d - loss: %.4f - val_loss: %.4f' % (
        epoch + 1, loss(y_train, out), loss(y_valid, val_out)))