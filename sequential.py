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
