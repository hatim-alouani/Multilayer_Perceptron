"""Multilayer perceptron: layers, forward pass, backpropagation, updates.

SHAPE CONVENTION -- one column per sample, everywhere:

    A_prev : (n_in,  m)      activations from the previous layer
    W      : (n_out, n_in)
    b      : (n_out, 1)
    Z      : (n_out, m)      pre-activation
    A      : (n_out, m)      post-activation
    Y      : (n_classes, m)  one-hot targets

`m` is the number of samples in the current batch.
"""

import numpy as np

EPS = 1e-15


# --------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------

def sigmoid(Z):
    """Numerically stable element-wise sigmoid."""
    out = np.empty_like(Z, dtype=np.float64)
    pos = Z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-Z[pos]))
    exp_z = np.exp(Z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def sigmoid_prime(A):
    """Derivative of sigmoid expressed in terms of its own output."""
    return A * (1.0 - A)


def relu(Z):
    return np.maximum(0.0, Z)


def relu_prime(A):
    return (A > 0.0).astype(np.float64)


def tanh(Z):
    return np.tanh(Z)


def tanh_prime(A):
    return 1.0 - A ** 2


def softmax(Z):
    """Column-wise softmax. Each column becomes a probability distribution."""
    shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_prime),
    "relu": (relu, relu_prime),
    "tanh": (tanh, tanh_prime),
    "softmax": (softmax, None),
}


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def categorical_cross_entropy(Y, A):
    """Mean categorical cross-entropy over a batch. Y and A are (n_classes, m)."""
    m = Y.shape[1]
    A = np.clip(A, EPS, 1.0 - EPS)
    return float(-np.sum(Y * np.log(A)) / m)


def binary_cross_entropy(y_true, p):
    """Subject's evaluation loss. y_true and p are 1-D arrays of length N."""
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


LOSSES = {
    "categoricalCrossentropy": categorical_cross_entropy,
    "binaryCrossentropy": categorical_cross_entropy,
}


# --------------------------------------------------------------------------
# Weight initialisation
# --------------------------------------------------------------------------

def he_uniform(n_out, n_in, rng):
    limit = np.sqrt(6.0 / n_in)
    return rng.uniform(-limit, limit, size=(n_out, n_in))


def he_normal(n_out, n_in, rng):
    return rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_out, n_in))


def xavier_uniform(n_out, n_in, rng):
    limit = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-limit, limit, size=(n_out, n_in))


INITIALIZERS = {
    "heUniform": he_uniform,
    "heNormal": he_normal,
    "xavierUniform": xavier_uniform,
}


# --------------------------------------------------------------------------
# Layer
# --------------------------------------------------------------------------

class DenseLayer:
    """One fully connected layer: parameters, activation, and cached values."""

    def __init__(self, n_in, n_out, activation="sigmoid",
                 weights_initializer="heUniform", rng=None):
        rng = np.random.default_rng() if rng is None else rng
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation '{activation}'")
        if weights_initializer not in INITIALIZERS:
            raise ValueError(f"unknown initializer '{weights_initializer}'")

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.W = INITIALIZERS[weights_initializer](n_out, n_in, rng)
        self.b = np.zeros((n_out, 1))

        self.A_prev = None
        self.Z = None
        self.A = None
        self.dW = None
        self.db = None

        # optimizer state (momentum / Adam)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)
        self.sW = np.zeros_like(self.W)
        self.sb = np.zeros_like(self.b)

    def activate(self, Z):
        return ACTIVATIONS[self.activation][0](Z)

    def activation_prime(self):
        prime = ACTIVATIONS[self.activation][1]
        if prime is None:
            raise ValueError(
                f"'{self.activation}' has no element-wise derivative; it is "
                "only valid as an output layer paired with cross-entropy"
            )
        return prime(self.A)

    def __repr__(self):
        return f"DenseLayer({self.n_in} -> {self.n_out}, {self.activation})"


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

class Network:
    """A stack of dense layers with a softmax output."""

    def __init__(self, layer_sizes, activation="sigmoid",
                 output_activation="softmax",
                 weights_initializer="heUniform",
                 optimizer="sgd", momentum=0.9,
                 beta1=0.9, beta2=0.999, seed=None):
        if len(layer_sizes) < 4:
            raise ValueError(
                "the subject requires at least two hidden layers, so "
                "layer_sizes needs at least 4 entries"
            )
        rng = np.random.default_rng(seed)
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self._t = 0

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            is_last = (i == len(layer_sizes) - 2)
            self.layers.append(DenseLayer(
                layer_sizes[i],
                layer_sizes[i + 1],
                activation=output_activation if is_last else activation,
                weights_initializer=weights_initializer,
                rng=rng,
            ))

        # preprocessing statistics, saved with the model
        self.mean = None
        self.std = None

    # ---- forward -----------------------------------------------------
    def forward(self, X):
        """Propagate X (n_features, m) through every layer, return the output."""
        A = X
        for layer in self.layers:
            layer.A_prev = A
            layer.Z = layer.W @ A + layer.b
            layer.A = layer.activate(layer.Z)
            A = layer.A
        return A

    # ---- backward ----------------------------------------------------
    def backward(self, Y):
        """Fill in dW and db on every layer. Y is (n_classes, m) one-hot."""
        m = Y.shape[1]
        last = len(self.layers) - 1
        dZ = None

        for i in range(last, -1, -1):
            layer = self.layers[i]
            if i == last:
                # softmax + cross-entropy collapse to this
                dZ = layer.A - Y
            else:
                dA = self.layers[i + 1].W.T @ dZ
                dZ = dA * layer.activation_prime()
            layer.dW = (dZ @ layer.A_prev.T) / m
            layer.db = np.sum(dZ, axis=1, keepdims=True) / m

    # ---- parameter update -------------------------------------------
    def update(self, learning_rate):
        self._t += 1
        for layer in self.layers:
            if self.optimizer == "sgd":
                layer.W -= learning_rate * layer.dW
                layer.b -= learning_rate * layer.db

            elif self.optimizer == "momentum":
                layer.vW = self.momentum * layer.vW + layer.dW
                layer.vb = self.momentum * layer.vb + layer.db
                layer.W -= learning_rate * layer.vW
                layer.b -= learning_rate * layer.vb

            elif self.optimizer == "adam":
                b1, b2, t = self.beta1, self.beta2, self._t
                layer.vW = b1 * layer.vW + (1 - b1) * layer.dW
                layer.vb = b1 * layer.vb + (1 - b1) * layer.db
                layer.sW = b2 * layer.sW + (1 - b2) * layer.dW ** 2
                layer.sb = b2 * layer.sb + (1 - b2) * layer.db ** 2
                vW_hat = layer.vW / (1 - b1 ** t)
                vb_hat = layer.vb / (1 - b1 ** t)
                sW_hat = layer.sW / (1 - b2 ** t)
                sb_hat = layer.sb / (1 - b2 ** t)
                layer.W -= learning_rate * vW_hat / (np.sqrt(sW_hat) + 1e-8)
                layer.b -= learning_rate * vb_hat / (np.sqrt(sb_hat) + 1e-8)

            else:
                raise ValueError(f"unknown optimizer '{self.optimizer}'")

    # ---- persistence -------------------------------------------------
    def save(self, path):
        payload = {
            "layer_sizes": self.layer_sizes,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "weights": [layer.W for layer in self.layers],
            "biases": [layer.b for layer in self.layers],
            "mean": self.mean,
            "std": self.std,
        }
        np.save(path, payload, allow_pickle=True)

    @classmethod
    def load(cls, path):
        payload = np.load(path, allow_pickle=True).item()
        net = cls(payload["layer_sizes"],
                  activation=payload["activation"],
                  output_activation=payload["output_activation"])
        for layer, W, b in zip(net.layers, payload["weights"], payload["biases"]):
            layer.W = W
            layer.b = b
        net.mean = payload["mean"]
        net.std = payload["std"]
        if net.mean is None or net.std is None:
            raise ValueError("model file has no normalization statistics")
        return net


# --------------------------------------------------------------------------
# Gradient checking
# --------------------------------------------------------------------------

def gradient_check(net, X, Y, epsilon=1e-6, max_checks=40):
    """Compare analytic gradients against finite differences."""
    net.forward(X)
    net.backward(Y)

    rng = np.random.default_rng(0)
    worst = 0.0
    checked = 0
    per_layer = max(1, max_checks // len(net.layers))

    for layer in net.layers:
        picks = rng.choice(layer.W.size, size=min(per_layer, layer.W.size),
                           replace=False)
        for flat in picks:
            i, j = np.unravel_index(flat, layer.W.shape)
            original = layer.W[i, j]

            layer.W[i, j] = original + epsilon
            loss_plus = categorical_cross_entropy(Y, net.forward(X))
            layer.W[i, j] = original - epsilon
            loss_minus = categorical_cross_entropy(Y, net.forward(X))
            layer.W[i, j] = original

            numeric = (loss_plus - loss_minus) / (2 * epsilon)
            analytic = layer.dW[i, j]
            denom = max(1e-12, abs(numeric) + abs(analytic))
            worst = max(worst, abs(numeric - analytic) / denom)
            checked += 1

    # recompute clean gradients, the perturbations left stale caches
    net.forward(X)
    net.backward(Y)

    ok = worst < 1e-5
    print(f"gradient check: {checked} weights, worst relative error "
          f"{worst:.3e} -> {'OK' if ok else 'MISMATCH'}")
    return ok
