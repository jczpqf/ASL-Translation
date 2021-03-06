"""
Converts jpg images to a single channel numpy arrays
Run this file from the directory where the image folder is contained
"""

import os
import argparse
import numpy as np
from PIL import Image
from utility import sample

parser = argparse.ArgumentParser(description='Parse potential arguments')
parser.add_argument(
    '--image_dir', type=str, help='the directory of the images',
    default='raw_images')
parser.add_argument(
    '--new_dir', type=str, help='the new director of the new images',
    default='grey_images')
parser.add_argument(
    '--sub_dir_labels', type=str, default='True', choices=['True', 'False']
)
parser.add_argument('--new_dim', type=int, default=25)
parser.add_argument('--sample_percent', type=float, default=1.0)

args = parser.parse_args()

image_dir = args.image_dir
new_dir = args.new_dir
sub_dir_labels = args.sub_dir_labels == 'True'
new_dim = args.new_dim
sample_percent = args.sample_percent


def remove_extension(file_name):
    """
    Removes everything at and after the first '.' character
    """
    for i, c in enumerate(file_name):
        if c == '.':
            return file_name[:i]
    return file_name


def transform(img):
    """
    transforms the images to greyscale downscaled image and transforms
    it into a numpy array
    """
    img = img.convert('L')  # convert image to greyscale
    img = img.resize((new_dim, new_dim), Image.ANTIALIAS)
    img = np.array(img)  # convert image to numpy array
    return img


root_dir = os.getcwd()
try:
    os.mkdir(root_dir + '\\' + new_dir)
except FileExistsError:  # if directory already exists thats ok
    pass
if sub_dir_labels:
    path_labels = os.listdir(root_dir + '\\' + image_dir)
    for label in path_labels:
        try:
            os.mkdir(root_dir + '\\' + new_dir + '\\' + label)
        except FileExistsError:  # if directory already exists thats ok
            pass
        files = os.listdir(root_dir + '\\' + image_dir + '\\' + label)
        files = sample(files, sample_percent)
        for file in files:
            img = Image.open(root_dir + '\\' + image_dir + '\\' +
                             label + '\\' + file)  # open image
            img = transform(img)
            np.save(root_dir + '\\' + new_dir + '\\' +
                    label + '\\' + remove_extension(file), img)
else:
    files = os.listdir(root_dir + '\\' + image_dir)
    files = sample(files, sample_percent)
    for file in files:
        # open image
        img = Image.open(root_dir + '\\' + image_dir + '\\' + file)

        img = transform(img)
        np.save(root_dir + '\\' + new_dir + '\\' + remove_extension(file), img)
