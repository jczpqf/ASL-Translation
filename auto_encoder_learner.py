import torch
import os
import numpy as np
from utility import load_data
from auto_encoder import AutoEncoder
import argparse


parser = argparse.ArgumentParser(description='Parse potential arguments')
parser.add_argument(
    '--image_dir', type=str, help='the directory of the image',
    default='grey_images_25')
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--save_name', type=str, default=None)
parser.add_argument('--sample_percent', type=float, default=1.0)
parser.add_argument('--layers', type=str, default='100')
parser.add_argument('--feature_output_path', type=str, default=None)

args = parser.parse_args()
load_path = args.load_path
image_dir = args.image_dir
batch_size = args.batch_size
epochs = args.epochs
save_name = args.save_name
sample_percent = args.sample_percent
feature_output_path = args.feature_output_path
layers = list(map(int, args.layers.split(',')))

X, labels, file_names = load_data(image_dir, sample_percent=sample_percent,
                                  return_names=True)
length, width = X[0].shape
input_size = length*width
X = torch.Tensor(X).view(-1, input_size).type(torch.float32)
X /= 255

if load_path is None:
    model = AutoEncoder([input_size] + layers)
    model.train(X=X, batch_size=batch_size, epochs=epochs, verbose=True)
else:
    model = AutoEncoder.load(load_path)

if feature_output_path is not None:
    print('Saving learned features...')
    new_features = model(X)
    new_features = new_features.detach().numpy()
    root_dir = os.getcwd()
    try:
        os.mkdir(root_dir + '\\' + feature_output_path)
    except FileExistsError:  # if directory already exists thats ok
        pass
    for label in np.unique(labels):
        try:
            os.mkdir(root_dir + '\\' + feature_output_path + '\\' + label)
        except FileExistsError:  # if directory already exists thats ok
            pass
    for x_data, label, file_name in zip(X, labels, file_names):
        np.save(root_dir + '\\' + feature_output_path + '\\' +
                label + '\\' + file_name, x_data)
