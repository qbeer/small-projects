import numpy as np


class Model:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, x, y, lr=0.001, EPOCHS=5):
        for epoch in range(EPOCHS):
            for ind in range(x.shape[0]):
                yhat = self._forward_pass(x[ind])
                loss = np.sum((y[ind] - yhat)**2)
                grad = -2*(y[ind] - yhat)
                grad = self._backward_pass(grad)
                self._apply_grads(lr)
                print('Loss in epoch %d, iteration %d : %.5f' % (epoch, ind, loss))

    def _forward_pass(self, x):
        input_ = x
        for layer in self.layers:
            input_ = layer.forward_pass(input_)
        return input_

    def _backward_pass(self, grad):
        for layer in self.layers[::-1]:
            grad = layer.backprop(grad)
        return grad

    def _apply_grads(self, lr):
        for layer in self.layers:
            layer.apply_gradients(lr)

    def predict(self, X):
        return np.array([self._forward_pass(test)[0] for test in X])
