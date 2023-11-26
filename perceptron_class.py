import numpy as np

class Perceptron(object):
    def __init__(self, eta=0.1, epochs=50):
        self.eta = eta
        self.epochs = epochs
        self.w_ = None

    def set_weights(self, weights):
        self.w_ = weights

    def train(self, X, y):
        if self.w_ is None:
            self.w_ = np.random.rand(1 + X.shape[1])

        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (self.predict(xi) - target)
                self.w_[:-1] -= update * xi
                self.w_[-1] -= update
                errors += int(update != 0)
            if errors == 0:
                return self
            else:
                self.errors_.append(errors)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[:-1]) + self.w_[-1]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)