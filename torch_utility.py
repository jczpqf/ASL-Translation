"""
functions that make life easier specific to pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


def flatten(input):
    return input.view(input.shape[0], -1)


class MLP(nn.Module):
    def __init__(self, design):
        super(MLP, self).__init__()
        self.design = design
        self.layers = nn.ModuleList()
        for i in range(len(design) - 1):
            self.layers.append(nn.Linear(design[i], design[i + 1]))

    def forward(self, x):
        if len(x) >= 3:
            x = flatten(x)
        for layer in self.layers[:len(self.layers) - 1]:
            x = F.relu(layer(x))
        x = self.layers[len(self.layers) - 1](x)
        if len(x.shape) == 1:
            return F.softmax(x, 0)
        return F.softmax(x, 1)

    @property
    def weight(self):
        return [layer.weight for layer in self.layers]

    @weight.setter
    def weight(self, v):
        raise AttributeError(str(type(self)) + ".weight cannot be modifed")

    def save(self, name):
        torch.save(self.state_dict(), name + '.weights')
        pickle.dump(self.design, open(name + '.params', 'wb'))

    @staticmethod
    def load(name):
        params = pickle.load(open(name + '.params', 'rb'))
        mlp = MLP(params)
        mlp.load_state_dict(torch.load(name + '.weights'))
        return mlp


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
    output = model(X)
    num_matches = torch.sum(torch.argmax(output, 1) == torch.argmax(y, 1))
    return float(num_matches) / len(X)
