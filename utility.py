"""
contains misc functions that make life easier
"""

import os
import numpy as np


def load_data(data_path, mode='r'):
    """
    default mode is read only
    """
    data_dict = {}
    label_dirs = os.listdir(data_path)
    for label in label_dirs:
        data = []
        label_data = os.listdir(data_path + '\\' + label)
        for file in label_data:
            data.append(np.load(
                data_path + '\\' + label + '\\' + file, mode=mode))
        data_dict[label] = data
    return data_dict
