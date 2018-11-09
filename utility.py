"""
contains misc functions that make life easier
"""

import os
import numpy as np
import random


def load_data(data_path, mode='r', sample_percent=1.0):
    """
    default mode is read only
    """
    X = []
    y = []
    label_dirs = os.listdir(data_path)
    for label in label_dirs:
        label_data = os.listdir(data_path + '\\' + label)
        label_data = sample(label_data, sample_percent)
        y += [label] * len(label_data)
        for file in label_data:
            X.append(np.load(data_path + '\\' + label + '\\' +
                             file, mmap_mode=mode))
    return np.array(X), np.array(y)


def one_hot(one_index, size):
    z = np.zeros(size)
    z[one_index] = 1
    return z


def batch_training_generator(X, y, batch_size):
    last_index = 0
    for i in range(batch_size, len(X), batch_size):
        yield X[last_index:i], y[last_index:i]
        last_index = i
    yield X[last_index:], y[last_index:]


def sample(pop, sample_percent):
    if sample_percent == 1.0:
        return pop
    sample_size = int(len(pop) * sample_percent)
    return random.sample(pop, sample_size)


def train_test_split(X, y, test_size=0.1):
    N = len(X)
    test_set_size = int(N * test_size)
    test_indices = random.sample(range(0, N), test_set_size)
    random.shuffle(test_indices)
    X_test, y_test = X[test_indices], y[test_indices]
    train_indicies = list(set(range(0, N)) - set(test_indices))
    random.shuffle(train_indicies)
    X_train, y_train = X[train_indicies], y[train_indicies]
    return X_train, X_test, y_train, y_test
