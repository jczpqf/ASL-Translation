import torch
import torch.nn as nn
from utility import train_test_split
from numpy import unique
from utility import load_data, batch_training_generator
from torch_utility import accuracy, Sequential, fully_connected
import argparse


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
X = torch.Tensor(X).view(-1, length*width).type(torch.float32)
X /= 255
unique_y = unique(labels)
num_unique_labels = len(unique_y)
mapping = {label: i for i, label in enumerate(unique_y)}

labels = torch.Tensor([mapping[label] for label in labels]).type(torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1,
                                                    shuffle=False)

N = len(X_train)
layers = fully_connected([length*width, 256, 256, 256, num_unique_labels])
model = Sequential(*layers)
op = torch.optim.Adam(model.parameters(), lr=.001)
loss_fn = nn.CrossEntropyLoss(reduction='sum')
for epoch_num in range(epochs):
    epoch_loss = 0
    training = batch_training_generator(X_train, y_train, batch_size,
                                        shuffle=True)
    for x_set, y_set in training:
        op.zero_grad()
        output = model(x_set)
        loss = loss_fn(output, y_set)
        loss /= N
        epoch_loss += float(loss)
        loss.backward()
        op.step()
    print(epoch_num, epoch_loss, accuracy(model, X_train, y_train))
model.append(nn.Softmax(dim=1))
print(accuracy(model, X_test, y_test))
if save_name is not None:
    model.save(save_name)
