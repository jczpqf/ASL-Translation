import torch
from utility import load_data
from auto_encoder import AutoEncoder
import argparse


parser = argparse.ArgumentParser(description='Parse potential arguments')
parser.add_argument(
    '--image_dir', type=str, help='the directory of the image',
    default='grey_images_25')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--save_name', type=str, default=None)
parser.add_argument('--sample_percent', type=float, default=1.0)
parser.add_argument('--layers', type=str, default='100')

args = parser.parse_args()
image_dir = args.image_dir
batch_size = args.batch_size
epochs = args.epochs
save_name = args.save_name
sample_percent = args.sample_percent
layers = list(map(int, args.layers.split(',')))

X, _ = load_data(image_dir, sample_percent=sample_percent)
length, width = X[0].shape
input_size = length*width
X = torch.Tensor(X).view(-1, input_size).type(torch.float32)
X /= 255

model = AutoEncoder([input_size] + layers)
model.train(X=X, batch_size=batch_size, epochs=epochs, verbose=True)
