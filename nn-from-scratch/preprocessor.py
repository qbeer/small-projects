import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def acquire_data():
    with open('scramtex_700_28px.pkl', 'rb') as f:
        data = pickle.load(f)

    X_train = data['train_images']
    X_test = data['test_images']
    y_train = data['train_labels']
    y_test = data['test_labels']

    X_train = np.clip(X_train, 0, 1.0)
    X_test = np.clip(X_test, 0, 1.0)

    mean, std = np.mean(X_train), np.std(X_train)

    print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    print('Min: %.3f, Max: %.3f' % (np.min(X_train), np.max(X_train)))
    print('Training size : %d \t Test size : %d\n\n' %
          (X_train.shape[0], X_test.shape[0]))

    X_train, _, y_train, _ = train_test_split(X_train.reshape(
        X_train.shape[0], 28*28), y_train, test_size=0.001, random_state=45)
    X_test, _, y_test, _ = train_test_split(X_test.reshape(
        X_test.shape[0], 28*28), y_test, test_size=0.001, random_state=42)

    return (X_train, y_train), (X_test, y_test)
