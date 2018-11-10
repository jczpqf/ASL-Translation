"""
functions that make life easier specific to pytorch
"""
import torch
import torch.nn as nn
import pickle


class Sequential(nn.Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.args = args
        self.layers = nn.ModuleList(self.args)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def pop(self):
        self.layers = self.layers[:len(self.layers) - 1]

    def append(self, layer):
        self.layers.append(layer)

    @property
    def weight(self):
        return [layer.weight for layer in self.layers
                if type(layer) is nn.Linear]

    @weight.setter
    def weight(self, v):
        raise AttributeError(str(type(self)) + ".weight cannot be modifed")

    def save(self, name):
        torch.save(self.state_dict(), name + '.weights')
        pickle.dump(self.args, open(name + '.params', 'wb'))

    @staticmethod
    def load(name):
        params = pickle.load(open(name + '.params', 'rb'))
        model = Sequential(*params)
        model.load_state_dict(torch.load(name + '.weights'))
        return model


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


def accuracy(model, X, y):
    """
    Calculates the accuracy of the model
    """
    X, y = check_convert(X), check_convert(y)
    if len(y.shape) == 2:
        y = torch.argmax(y, 1)
    output = model(X)
    num_matches = torch.sum(torch.argmax(output, 1) == y)
    return float(num_matches) / len(X)
