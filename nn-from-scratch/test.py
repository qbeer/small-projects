from dense_layer import DenseLayer
from activation import Relu, Sigmoid
import numpy as np

import pickle
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from model import Model
from keras.utils import to_categorical
from preprocessor import acquire_data
from keras.datasets.mnist import load_data

(x, y), (x_test, y_test) = acquire_data()

seq = Model()
seq.add(DenseLayer(784, 512, Relu()))
seq.add(DenseLayer(512, 256, Relu()))
seq.add(DenseLayer(256, 64, Relu()))
seq.add(DenseLayer(64, 1, Relu()))

seq.fit(x, y, lr=5e-6, EPOCHS=250)

pred_test = seq.predict(x_test)

print("\nAccuracy : ", accuracy_score(pred_test.astype(int), y_test))