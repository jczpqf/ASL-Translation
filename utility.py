"""
contains misc functions that make life easier
"""

import os
import numpy as np


def load_data(data_path, mode='r'):
    """
    default mode is read only
    """
    X = []
    y = []
    label_dirs = os.listdir(data_path)
    for label in label_dirs:
        label_data = os.listdir(data_path + '\\' + label)
        y += [label] * len(label_data)
        for file in label_data:
            X.append(np.load(data_path + '\\' + label + '\\' +
                             file, mmap_mode=mode))
    return np.array(X), np.array(y)
