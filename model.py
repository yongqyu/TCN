from collections import namedtuple
import numpy as np
import functools
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchvision
from torchvision import models
#==================================Function====================================/
#==============================================================================/

def weights_init_normal(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('Linear') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
	#print('initialization method [%s]' % init_type)
	if init_type == 'normal':
		net.apply(weights_init_normal)
	elif init_type == 'xavier':
		net.apply(weights_init_xavier)
	elif init_type == 'kaiming':
		net.apply(weights_init_kaiming)
	elif init_type == 'orthogonal':
		net.apply(weights_init_orthogonal)
	else:
		raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	elif norm_type == 'none':
		norm_layer = None
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer

#==================================ResnetB=====================================/
#==============================================================================/

# Define a resnet block
class ResnetBlock(nn.Module):
	def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		super(ResnetBlock, self).__init__()
		self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

	def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		conv_block = []
		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)

		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
					   norm_layer(dim),
					   nn.ReLU(True)]
		if use_dropout:
			conv_block += [nn.Dropout(0.5)]

		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
					   norm_layer(dim)]

		return nn.Sequential(*conv_block)

	def forward(self, x):
		out = x + self.conv_block(x)
		return out

#==================================Encoder=====================================/
#==============================================================================/

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class Resnet_Encoder(nn.Module):
	def __init__(self, num_classes, input_nc=2, ngf=32, norm_layer=get_norm_layer(norm_type='batch'),
					   embedding_size=256, use_dropout=False, n_blocks=6, padding_type='reflect'):
		assert(n_blocks >= 0)
		super(Resnet_Encoder, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		model = [nn.ReflectionPad2d(3),
				 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
						   bias=use_bias),
				 norm_layer(ngf),
				 nn.LeakyReLU(0.01, inplace=True)]

		n_downsampling = 3
		for i in range(n_downsampling):
			mult = 2**i
			model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=7,
								stride=3, padding=2, bias=use_bias),
					  norm_layer(ngf * mult * 2),
					  nn.LeakyReLU(0.01, inplace=True)]

		mult = 2**n_downsampling
		for i in range(n_blocks):
			model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		self.model = nn.Sequential(*model)
		self.classifier = nn.Sequential(
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(ngf * mult * 16, num_classes)
		)
		init_weights(self.model)
		init_weights(self.classifier)

	def forward(self, input):
		vec = self.model(input)
		vec = vec.view(vec.size(0), -1)
		cls = self.classifier(vec)
		return vec, cls

#===============================Discriminator==================================/
#==============================================================================/

class Resnet_Discriminator(nn.Module):
	def __init__(self, tf_classes=1, st_classes=150, ch_classes=1000, input_nc=2, ngf=32,
						norm_layer=get_norm_layer(norm_type='batch'),
						use_dropout=False, n_blocks=6, padding_type='reflect'):
		assert(n_blocks >= 0)
		super(Resnet_Discriminator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		model = [nn.ReflectionPad2d(3),
				 nn.Conv2d(input_nc, ngf, kernel_size=5, padding=0,
						   bias=use_bias),
				 norm_layer(ngf),
				 nn.LeakyReLU(0.01, inplace=True)]

		n_downsampling = 3
		for i in range(n_downsampling):
			mult = 2**i
			model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=7,
								stride=3, padding=2, bias=use_bias),
					  norm_layer(ngf * mult * 2),
					  nn.LeakyReLU(0.01, inplace=True)]

		mult = 2**n_downsampling
		for i in range(n_blocks):
			model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		self.model = nn.Sequential(*model)
		self.tf_classifier = nn.Conv2d(ngf * mult, tf_classes, kernel_size=4, stride=1, padding=0, bias=False)
		self.st_classifier = nn.Conv2d(ngf * mult, st_classes, kernel_size=4, stride=1, padding=0, bias=False)
		self.ch_classifier = nn.Conv2d(ngf * mult, ch_classes, kernel_size=4, stride=1, padding=0, bias=False)
		init_weights(self.model)
		init_weights(self.tf_classifier)
		init_weights(self.st_classifier)
		init_weights(self.ch_classifier)

	def forward(self, input):
		vec = self.model(input)
		vec = F.tanh(vec)
		tf_cls = self.tf_classifier(vec)
		st_cls = self.st_classifier(vec)
		ch_cls = self.ch_classifier(vec)
		return tf_cls.squeeze(), st_cls.squeeze(), ch_cls.squeeze()

#=================================Generator====================================/
#==============================================================================/

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
	"""Custom deconvolutional layer for simplicity."""
	layers = []
	layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
	if bn:
		layers.append(nn.BatchNorm2d(c_out))
	return nn.Sequential(*layers)


class Generator(nn.Module):
	"""Generator containing 7 deconvolutional layers."""
	def __init__(self, c_in=512, c_out=2, image_size=64, conv_dim=64):
		super(Generator, self).__init__()
		self.mixer = nn.Conv2d(c_in, c_in, 1)
		self.fc = deconv(c_in, conv_dim*8, 5, 1, 0, bn=False)
		self.deconv1 = deconv(conv_dim*8, conv_dim*4, 4)
		self.deconv2 = deconv(conv_dim*4, conv_dim*2, 4)
		self.deconv3 = deconv(conv_dim*2, conv_dim, 4)
		self.deconv4 = deconv(conv_dim, c_out, 4, bn=False)

	def forward(self, c_org, style, char, c_trg):
		style = style.view(style.size(0), -1, 4, 4)
		char = char.view(char.size(0), -1, 4, 4)
		c_org = c_org.view(c_org.size(0), c_org.size(1), 1, 1)
		c_org = c_org.repeat(1, 1, char.size(2), char.size(3))
		c_trg = c_trg.view(c_trg.size(0), c_trg.size(1), 1, 1)
		c_trg = c_trg.repeat(1, 1, char.size(2), char.size(3))
		z = torch.cat([c_org, style, char, c_trg], dim=1)
		z = self.mixer(z)
		out = self.fc(z)
		out = F.leaky_relu(self.deconv1(out), 0.05)
		out = F.leaky_relu(self.deconv2(out), 0.05)
		out = F.leaky_relu(self.deconv3(out), 0.05)
		out = F.tanh(self.deconv4(out))
		return out
