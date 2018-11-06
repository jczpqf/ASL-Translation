import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from numpy import unique
from utility import load_data, one_hot, batch_training_generator
from torch_utility import accuracy
import numpy as np
import argparse


class NN(nn.Module):
    def __init__(self, design):
        super(NN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(design) - 1):
            self.layers.append(nn.Linear(design[i], design[i + 1]))

    def forward(self, x):
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


parser = argparse.ArgumentParser(description='Parse potential arguments')
parser.add_argument(
    '--image_dir', type=str, help='the directory of the image',
    default='grey_images_25')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--save_name', type=str, default=None)
parser.add_argument('--sample_percent', type=float, default=1.0)

args = parser.parse_args()
image_dir = args.image_dir
batch_size = args.batch_size
epochs = args.epochs
save_name = args.save_name
sample_percent = args.sample_percent

X, labels = load_data(image_dir, sample_percent=sample_percent)
length, width = X[0].shape
X = X.reshape(-1, length * width)
X = X.astype('float32')
X /= 255
unique_y = unique(labels)
num_unique_labels = len(unique_y)
mapping = {label: one_hot(i, num_unique_labels)
           for i, label in enumerate(unique_y)}
y = np.array([mapping[label] for label in labels])
y = y.astype('float32')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)


model = NN([length * width, 512, 512, num_unique_labels])
op = torch.optim.Adam(model.parameters(), lr=.01)
loss_fn = nn.MSELoss()
for epoch_num in range(epochs):
    epoch_loss = 0
    training = batch_training_generator(X_train, y_train, batch_size)
    for x_set, y_set in training:
        op.zero_grad()
        output = model(x_set)
        loss = loss_fn(output, y_set)
        epoch_loss += float(loss)
        loss.backward()
        op.step()
    print(epoch_num, epoch_loss, accuracy(model, X_train, y_train))

print(accuracy(model, X_test, y_test))
if save_name is not None:
    torch.save(nn.state_dict(), save_name)
