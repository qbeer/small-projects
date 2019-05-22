from dense_layer import DenseLayer
from activation import Relu, Sigmoid
import numpy as np

import pickle
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from model import Model
from keras.utils import to_categorical
from preprocessor import acquire_data

seq = Model()
seq.add(DenseLayer(784, 10, Relu()))
seq.add(DenseLayer(10, 5, Relu()))
seq.add(DenseLayer(5, 2, Relu()))
seq.add(DenseLayer(2, 1, Relu()))

(x, y), (x_test, y_test) = acquire_data()

seq.fit(x, y, EPOCHS=5)

pred_test = seq.predict(x_test).astype(int)

print("Accuracy : ", accuracy_score(pred_test, y_test))