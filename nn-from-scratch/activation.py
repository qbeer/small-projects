import numpy as np

class Sigmoid:
    def call(self, x):
        return 1. / (1. + np.exp(-x))
    
    def derivative(self, x):
        deriv = np.diagflat(self.call(x) * ( 1. - self.call(x)))
        if deriv.shape == (1, 1): deriv = deriv.flatten()
        return deriv

class Softmax:
    def call(self, x):
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps/np.sum(exps)

    def derivative(self, x):
        softmax = self.call(x)
        deriv = np.diagflat(softmax)
        deriv = deriv - np.outer(softmax, softmax)
        return deriv

class Relu:
    def call(self, x):
        return np.maximum(np.zeros(x.shape[0]), x)

    def derivative(self, x):
        return np.array([0 if val < 0 else 1 for val in x])
