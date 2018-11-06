"""
functions that make life easier specific to pytorch
"""
import torch


def check_convert(e):
    if type(e) is not torch.Tensor:
        return torch.Tensor(e)
    return e


def accuracy(model, X, y):
    X, y = check_convert(X), check_convert(y)
    output = model(X)
    num_matches = torch.sum(torch.argmax(output, 1) == torch.argmax(y, 1))
    return float(num_matches) / len(X)
