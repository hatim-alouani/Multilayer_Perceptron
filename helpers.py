import pandas as pd
import numpy as np

weights = []
biases = []

def load(path):
    df = pd.read_csv(path, header=None)
    x = df.iloc[:, 2:].values.astype(float)
    y = df.iloc[:, 1].values
    return x, y

# activations
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# loss functions
def categorical_cross_entropy(y, l):
    l = np.clip(l, 1e-15, 1 - 1e-15)
    log_l = np.log(l)
    total_error = np.sum(y * log_l)
    return -total_error / len(y)

def binary_cross_entropy(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    total_error = np.sum((y * np.log(p)) + (1 - y) * np.log(1 - p))
    return -total_error / len(y)

def forward(x):
    l = x
    l_list = [x]
    for i in range(len(weights)):
        z = l @ weights[i] + biases[i]
        if i == len(weights) - 1:
            l = softmax(z)
        else:
            l = sigmoid(z)
        l_list.append(l)
    return l, l_list

def backward(x, y, l_list):
    m = len(x)
    dW = []
    db = []

    delta = l_list[-1] - y

    for i in range(len(weights) - 1, -1, -1):
        dw = l_list[i].T @ delta / m
        dbias = np.sum(delta, axis=0, keepdims=True) / m
        dW.insert(0, dw)
        db.insert(0, dbias)
        if i > 0:
            delta = (delta @ weights[i].T) * l_list[i] * (1 - l_list[i])

    return dW, db

def gradient_descent(weights, biases, dW, db, learning_rate):
    for i in range(len(weights)):
        weights[i] = weights[i] - learning_rate * dW[i]
        biases[i]  = biases[i]  - learning_rate * db[i]
    return weights, biases

def accuracy(y, l):
    predicted_class = np.argmax(l, axis=1)
    true_class = np.argmax(y, axis=1)
    correct_predictions = np.sum(predicted_class == true_class)
    total_predictions = len(y)
    return correct_predictions / total_predictions