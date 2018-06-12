#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image, ImageOps, ImageFont, ImageDraw


class ImageFolder(data.Dataset):
	"""Load Variaty Chinese Fonts for Iterator. """
	def __init__(self, root, transform=None, image_size=128):
		"""Initializes image paths and preprocessing module."""
		self.root = root
		self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
		self.transform = transform
		self.image_size = image_size
		print("image count in path :", len(self.image_paths))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.image_paths[index]
		style_idx = int(image_path.split('_')[0][len(self.root):])
		char_idx = int(image_path.split('_')[1][:-len(".png")])

		target_path = random.choice([x for x in self.image_paths
									if str(style_idx)+'_' in x and '_'+str(char_idx) not in x])
		style_trg_idx = int(target_path.split('_')[0][len(self.root):])
		char_trg_idx = int(target_path.split('_')[1][:-len(".png")])

		# Image split
		image = Image.open(image_path)
		inverted_image = ImageOps.invert(image)
		target = Image.open(target_path)
		inverted_target = ImageOps.invert(target)

		src_img = np.reshape(image.crop((0,0,self.image_size,self.image_size)),(128,128,1))
		trg_img = np.reshape(target.crop((0,0,self.image_size,self.image_size)),(128,128,1))
		inverted_src_img = np.reshape(inverted_image.crop((0,0,self.image_size,self.image_size)),(128,128,1))
		inverted_trg_img = np.reshape(inverted_target.crop((0,0,self.image_size,self.image_size)),(128,128,1))
		src_img = np.concatenate((src_img,inverted_src_img),axis=2)
		trg_img = np.concatenate((trg_img,inverted_trg_img),axis=2)

		if self.transform is not None:
			src_img = self.transform(src_img)
			trg_img = self.transform(trg_img)
		return src_img, trg_img, style_idx, char_idx, char_trg_idx, style_trg_idx

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)

class EncImageFolder(data.Dataset):
    """Load Variaty Styles for Iterator. """
    def __init__(self, root, transform=None, image_size=128):
        """Initializes image paths and preprocessing module."""
        self.root = root
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.transform = transform
        self.image_size = image_size
        print("image count in path :", len(self.image_paths))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        style_idx = int(image_path.split('_')[0][len(self.root):])
        char_idx = int(image_path.split('_')[1])
        char_tg_idx = int(image_path.split('_')[2])
        style_tg_idx = int(image_path.split('_')[3][:-len(".png")])

        # Image split
        image = Image.open(image_path)
        inverted_image = ImageOps.invert(image)

        src_img = np.reshape(image.crop((0,0,self.image_size,self.image_size)),(128,128,1))
        char_img = np.reshape(image.crop((self.image_size,0,2*self.image_size,self.image_size)),(128,128,1))
        style_img = np.reshape(image.crop((2*self.image_size,0,3*self.image_size,self.image_size)),(128,128,1))

        inverted_src_img = np.reshape(inverted_image.crop((0,0,self.image_size,self.image_size)),(128,128,1))
        inverted_char_img = np.reshape(inverted_image.crop((self.image_size,0,2*self.image_size,self.image_size)),(128,128,1))
        inverted_style_img = np.reshape(inverted_image.crop((2*self.image_size,0,3*self.image_size,self.image_size)),(128,128,1))

        src_img = np.concatenate((src_img,inverted_src_img),axis=2)
        char_img = np.concatenate((char_img,inverted_char_img),axis=2)
        style_img = np.concatenate((style_img,inverted_style_img),axis=2)

        if self.transform is not None:
            src_img = self.transform(src_img)
            char_img = self.transform(char_img)
            style_img = self.transform(style_img)
        return src_img, char_img, style_img, style_idx, char_idx, char_tg_idx, style_tg_idx

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2):
	"""Builds and returns Dataloader."""

	transform = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	dataset = ImageFolder(image_path, transform, image_size)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader
