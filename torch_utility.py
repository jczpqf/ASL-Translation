"""
functions that make life easier specific to pytorch
"""
import torch
import torch.nn as nn
from utility import batch_training_generator


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


def fully_connected(layout, activation=nn.ReLU(), last_layer_activation=None):
    lst = []
    for i in range(len(layout) - 2):
        lst.append(nn.Linear(layout[i], layout[i + 1]))
        lst.append(activation)
    lst.append(nn.Linear(layout[len(layout) - 2], layout[len(layout) - 1]))
    if last_layer_activation is not None:
        lst.append(last_layer_activation)
    return lst


def check_convert(item):
    """
    Checks if  item is a torch.Tensor and if it isn't turns it into one
    """
    if type(item) is not torch.Tensor:
        return torch.Tensor(item)
    return item


def accuracy(model, X, y, batch_size=32):
    """
    Calculates the accuracy of the model
    """
    X, y = check_convert(X), check_convert(y)
    if len(y.shape) == 2:
        y = torch.argmax(y, 1)
    num_matches = 0
    batch_gen = batch_training_generator(X, y, batch_size, shuffle=False)
    for x_data, y_data in batch_gen:
        output = model(x_data)
        num_matches += int(torch.sum(torch.argmax(output, 1) == y_data))
    return num_matches / len(X)
