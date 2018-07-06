import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.measure import compare_ssim as ssim
from model import Resnet_Encoder
from model import Resnet_Discriminator
from model import Generator
import utils


class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.enc_style = None
		self.enc_char = None
		self.generator = None
		self.discriminator = None
		self.g_optimizer = None
		self.d_optimizer = None
		self.celoss = torch.nn.CrossEntropyLoss()

		# Model hyper-parameters
		self.z_dim = config.z_dim
		self.image_size = config.image_size
		self.d_train_repeat = config.d_train_repeat

		# Hyper-parameters
		self.lambda_cls = config.lambda_cls
		self.lambda_gp = config.lambda_gp
		self.lambda_rec = config.lambda_rec
		self.g_lr = config.g_lr
		self.d_lr = config.d_lr
		self.e_lr = config.e_lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.start_epochs = config.start_epochs
		self.sample_epochs = config.sample_epochs
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.enc_epochs = config.enc_epochs
		self.batch_size = config.batch_size
		self.style_cnt = config.style_cnt
		self.char_cnt = config.char_cnt

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.sample_path = config.sample_path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.build_model()

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def build_model(self):
		"""Build generator and discriminator."""
		self.enc_style = Resnet_Encoder(num_classes=self.style_cnt, embedding_size=self.z_dim)
		self.enc_char = Resnet_Encoder(num_classes=self.char_cnt, embedding_size=self.z_dim)
		self.decoder = Generator(c_in=2*self.z_dim + 2*self.char_cnt)
		self.discriminator = Resnet_Discriminator(tf_classes=1,
												st_classes=self.style_cnt, ch_classes=self.char_cnt)
		self.generator = Generator(c_in=2*self.z_dim + 2*self.char_cnt)
		self.e_optimizer = optim.Adam(list(self.enc_style.parameters()) + list(self.enc_char.parameters()) + \
									  list(self.decoder.parameters()),
									  self.e_lr, [self.beta1, self.beta2])
		self.g_optimizer = optim.Adam(list(self.generator.parameters()) + \
									  list(self.enc_style.parameters()) + list(self.enc_char.parameters()),
									  self.g_lr, [self.beta1, self.beta2])
		self.d_optimizer = optim.Adam(self.discriminator.parameters(),
									  self.d_lr, [self.beta1, self.beta2])


		if torch.cuda.is_available():
			self.enc_char.cuda()
			self.enc_style.cuda()
			self.decoder.cuda()
			self.generator.cuda()
			self.discriminator.cuda()
		self.print_network(self.enc_style, 'SE')
		self.print_network(self.enc_char, 'CE')
		self.print_network(self.discriminator, 'D')
		self.print_network(self.generator, 'G')

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def restore_model(self, resume_iters):
		"""Restore the trained generator and discriminator."""
		G_path = os.path.join(self.model_path, 'generator-{}.pkl'.format(resume_iters))
		D_path = os.path.join(self.model_path, 'discriminator-{}.pkl'.format(resume_iters))
		if os.path.isfile(G_path) and os.path.isfile(D_path):
			self.generator.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
			self.discriminator.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
			print('G/D is Successfully Loaded from %s'%G_path)
			return resume_iters
		else:
			return 0

	def to_variable(self, x):
		"""Convert tensor to variable."""
		if torch.cuda.is_available():
			x = x.cuda()
		return Variable(x)

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.g_optimizer.param_groups:
			param_group['lr'] = g_lr
		for param_group in self.d_optimizer.param_groups:
			param_group['lr'] = d_lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.enc_style.zero_grad()
		self.enc_char.zero_grad()
		self.generator.zero_grad()
		self.discriminator.zero_grad()

	def denorm(self, x):
		"""Convert range (-1, 1) to (0, 1)"""
		out = (x + 1) / 2
		return out.clamp(0, 1)

	def compute_accuracy(self, x, y):
		_, predicted = torch.max(x, dim=1)
		correct = (predicted == y).float()
		accuracy = torch.mean(correct) * 100.0
		return accuracy

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img

	def label2onehot(self, labels):
		"""Convert label indices to one-hot vectors."""
		batch_size = labels.size(0)
		out = torch.zeros(batch_size, self.char_cnt)
		out[np.arange(batch_size), labels.long()] = 1
		return out.to(self.device)

	def gradient_penalty(self, y, x):
		"""Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
		weight = torch.ones(y.size()).to(self.device)
		dydx = torch.autograd.grad(outputs=y,
								   inputs=x,
								   grad_outputs=weight,
								   retain_graph=True,
								   create_graph=True,
								   only_inputs=True)[0]

		dydx = dydx.view(dydx.size(0), -1)
		dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
		return torch.mean((dydx_l2norm-1)**2)


	def pretrain(self):
		"""Train encoder."""
		#====================================== Pretrain ===========================================#
		#===========================================================================================#
		se_path = os.path.join(self.model_path, 'style_encoder-%d.pkl' %(self.enc_epochs))
		ce_path = os.path.join(self.model_path, 'char_encoder-%d.pkl' %(self.enc_epochs))

		# Train for encoder
		e_lr = self.e_lr
		best_se_acc = 0.
		best_ce_acc = 0.
		for epoch in range(self.enc_epochs):

			#===================================== train e ====================================#
			style_acc = 0.
			char_acc = 0.
			self.enc_style.train(True)
			for i, (images, ch_tg, st_tg, real_style, real_char, targ_char, _) in enumerate(self.train_loader):

				images = self.to_variable(images)
				ch_tg = self.to_variable(ch_tg)
				st_tg = self.to_variable(st_tg)
				real_style = self.to_variable(real_style)
				real_char = self.to_variable(real_char)

				anchor_style, style = self.enc_style(images)
				anchor_char, char = self.enc_char(images)
				style_acc += self.compute_accuracy(style, real_style)
				char_acc += self.compute_accuracy(char, real_char)
				style_loss = self.celoss(style, real_style)
				char_loss = self.celoss(char, real_char)
				e_loss = style_loss + char_loss

				# triplet loss
				if (style_acc.data.cpu().numpy()/(i+1)) > 80.:
					positive,_ = self.enc_style(ch_tg)
					negative,_ = self.enc_style(st_tg)

					triplet_loss_style = F.triplet_margin_loss(anchor_style, positive, negative, margin=1)
					e_loss += triplet_loss_style

				if (char_acc.data.cpu().numpy()/(i+1)) > 80.:
					positive,_ = self.enc_char(st_tg)
					negative,_ = self.enc_char(ch_tg)

					triplet_loss_char = F.triplet_margin_loss(anchor_char, positive, negative)
					e_loss += triplet_loss_char

				real_char_onehot = self.label2onehot(real_char)

				rec_img = self.decoder(real_char_onehot, anchor_style, anchor_char, real_char_onehot)
				ae_loss = torch.mean(torch.abs(images - rec_img))
				e_loss += ae_loss

				# backprop + optimize
				self.reset_grad()
				e_loss.backward()
				self.e_optimizer.step()

			# print the log info
			print('epoch [%d/%d], style_loss: %.4f, style_acc: %.4f, ae_loss: %.4f' % (
				  epoch+1, self.enc_epochs, \
				  style_loss.item(), style_acc.data.cpu().numpy()/(i+1), \
				  ae_loss.item()))

			# decay learning rate
			if (epoch+1) > (self.enc_epochs - self.num_epochs_decay):
				e_lr -= (self.e_lr / float(self.num_epochs_decay))
				for param_group in self.e_optimizer.param_groups:
					param_group['lr'] = e_lr
				print ('decay learning rate to e_lr: {}.'.format(e_lr))

			#===================================== valid e ====================================#
			style_acc = 0.
			char_acc = 0.
			self.enc_style.train(False)
			self.enc_char.train(False)
			self.enc_style.eval()
			self.enc_char.eval()
			for i, (images, _, _, real_style, real_char, _, _) in enumerate(self.valid_loader):
				images = self.to_variable(images)
				real_style = self.to_variable(real_style)
				real_char = self.to_variable(real_char)

				_, style = self.enc_style(images)
				style_acc += self.compute_accuracy(style, real_style)
				_, char = self.enc_char(images)
				char_acc += self.compute_accuracy(char, real_char)

			# save best style encoder model
			if (style_acc.data.cpu().numpy()/(i+1)) > best_se_acc:
				best_se_acc = style_acc.data.cpu().numpy()/(i+1)
				# save the model parameters
				best_enc_style = self.enc_style.state_dict()
				print('style encoder valid accuracy : %.2f'%best_se_acc)
			if (char_acc.data.cpu().numpy()/(i+1)) > best_ce_acc:
				best_ce_acc = char_acc.data.cpu().numpy()/(i+1)
				# save the model parameters
				best_enc_char = self.enc_char.state_dict()
				print('char encoder valid accuracy : %.2f'%best_ce_acc)
		torch.save(best_enc_style, se_path)
		print('save style encoder at accuracy %.2f'%best_se_acc)
		del best_enc_style, best_se_acc
		torch.save(best_enc_char, ce_path)
		print('Save char encoder at accuracy %.2f'%best_ce_acc)
		del best_enc_char, best_ce_acc
		del self.valid_loader, self.decoder


	def train(self):
		"""Train generator and discriminator."""
		#======================================= Main-Train ======================================#
		#=========================================================================================#
		se_path = os.path.join(self.model_path, 'style_encoder-%d.pkl' %(self.enc_epochs))
		ce_path = os.path.join(self.model_path, 'char_encoder-%d.pkl' %(self.enc_epochs))

		# Load the pretrained Encoder
		self.enc_style.load_state_dict(torch.load(se_path))
		self.enc_char.load_state_dict(torch.load(ce_path))
		print('Enocder is Successfully Loaded from %s'%se_path)

		start_iter = self.restore_model(self.start_epochs)
		self.num_epochs += start_iter

		# Main G/D Train
		iters_per_epoch = len(self.train_loader)
		g_lr = self.g_lr
		d_lr = self.d_lr
		start_time = time.time()
		for epoch in range(start_iter, self.num_epochs):
			self.generator.train(True)
			self.discriminator.train(True)
			for i, (x_style, x_trg, style, char, x_trg_c, x_trg_s) in enumerate(self.train_loader):

				loss = {}
				batch_size = x_trg.size(0)
				# Generate real labels
				x_style = self.to_variable(x_style)
				style = self.to_variable(style)
				char = self.to_variable(char)
				char_onehot = self.label2onehot(char)
				# Character transfer. keep style. and thats' character index
				x_trg = self.to_variable(x_trg)
				x_trg_s = self.to_variable(x_trg_s)
				x_trg_c = self.to_variable(x_trg_c)
				x_trg_c_onehot = self.label2onehot(x_trg_c)

				#==================================== Train D ====================================#

 				# Generate fake image from Encoder
				real_style, _ = self.enc_style(x_style)
				real_char, _ = self.enc_char(x_style)
				fake_img = self.generator(char_onehot, real_style, real_char, x_trg_c_onehot)

				# 1) Train D to recognize real images as real.
				out_src, out_style, out_char = self.discriminator(x_style)
				d_loss_real = torch.mean((out_src - 1) ** 2)		# Least square GANs

				# 2) Traing D to classify G(E(i)) as correct style/char
				d_loss_style = self.celoss(out_style, style)
				d_loss_char = self.celoss(out_char, char)

				# 3) Train D to recognize fake images as fake.
				fake_src, _,_ = self.discriminator(fake_img)
				d_loss_fake  = torch.mean(fake_src ** 2)			# Least Square GANs

				# 4) Compute loss for gradient penalty.
				alpha = torch.rand(x_style.size(0), 1, 1, 1).to(self.device)
				x_hat = (alpha * x_style.data + (1 - alpha) * fake_img.data).requires_grad_(True)
				out_src,_,_ = self.discriminator(x_hat)
				d_loss_gp = self.gradient_penalty(out_src, x_hat)

				# Compute gradient penalty
				d_loss = d_loss_real + d_loss_fake + \
						 self.lambda_cls * (d_loss_style + d_loss_char) + \
						 self.lambda_gp * d_loss_gp

				# Logging
				loss['D/loss_real'] = d_loss_real.item()
				loss['D/loss_fake'] = d_loss_fake.item()
				loss['D/loss_style'] = d_loss_style.item()
				loss['D/loss_char'] = d_loss_char.item()
				loss['D/loss_gp'] = d_loss_gp.item()

				# Backward + Optimize
				self.reset_grad()
				d_loss.backward()
				self.d_optimizer.step()

				#==================================== Train G ====================================#
				if (i+1) % self.d_train_repeat == 0:

					# Generate fake image from Encoder
					real_style, _ = self.enc_style(x_style)
					real_char, _ = self.enc_char(x_style)

					fake_img = self.generator(char_onehot, real_style, real_char, x_trg_c_onehot)

					# Generate identity image from Encoder
					id_img = self.generator(char_onehot, real_style, real_char, char_onehot)

					# Reconstruct Image from fake image
					fake_style, _ = self.enc_style(fake_img)
					fake_char, _ = self.enc_char(fake_img)
					rec_img = self.generator(x_trg_c_onehot, fake_style, fake_char, char_onehot)

					rec_style, _ = self.enc_style(rec_img)
					rec_char, _ = self.enc_char(rec_img)

					# 1) Train G so that D recognizes G(E(i)) as real.
					out_src, out_style, out_char = self.discriminator(fake_img)
					g_loss_fake = torch.mean((out_src - 1) ** 2)

					# 2) Training G so that D classifies G(E(i)) as correct style/char
					g_loss_style = self.celoss(out_style, x_trg_s)
					g_loss_char = self.celoss(out_char, x_trg_c)

					# 3) Training G to G(E(G(E(i)))) and (i) are similar. Reconstruct Loss
					g_loss_rec = torch.mean(torch.abs(x_style - rec_img))

					# 3.5.2) Training G to E(G(E(G(E(i))))) and E(i) are similar. Recons_Perceptual Loss
					g_loss_per_style = torch.mean((real_style - rec_style) ** 2)
					g_loss_per_char = torch.mean((real_char - rec_char) ** 2)
					g_loss_rec_per = g_loss_per_style + g_loss_per_char

					# 4) Training G to G(E(i)) and (i) are similar. Identity loss
					g_loss_id = torch.mean(torch.abs(x_style - id_img))

					# 6) Training G to 'x_style' and G(E(i)) are similar. L1 loss
					g_loss_l1 = torch.mean(torch.abs(x_trg - fake_img))

					# Compute Structural similarity measure of the Generator
					ssim_fake = utils.ssim(x_trg, fake_img)

					# Compute gradient penalty
					g_loss = g_loss_fake + \
							 self.lambda_cls * (g_loss_style + g_loss_char) + \
					 	     self.lambda_rec * (g_loss_rec + g_loss_rec_per
					 	     + g_loss_l1 - ssim_fake) + g_loss_id

					# Backprop + Optimize
					self.reset_grad()
					g_loss.backward()
					self.g_optimizer.step()

					# Logging
					loss['G/loss_fake'] = g_loss_fake.item()
					loss['G/loss_style'] = g_loss_style.item()
					loss['G/loss_char'] = g_loss_char.item()
					loss['G/loss_rec'] = g_loss_rec.item()
					loss['G/loss_id'] = g_loss_id.item()
					loss['G/loss_l1'] = g_loss_l1.item()
					loss['G/loss_ssim'] = 1.-ssim_fake.item()

					# Print the log info
					if (i+1) % self.log_step == 0:
						elapsed = time.time() - start_time
						elapsed = str(datetime.timedelta(seconds=elapsed))

						log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
								elapsed, epoch+1, self.num_epochs, i+1, iters_per_epoch)

						for tag, value in loss.items():
							log += ", {}: {:.4f}".format(tag, value)
						print(log)

						# Save real images
						x_trg = x_trg.view(x_trg.size(0), 2, self.image_size, self.image_size)
						x_trg = self.tensor2img(x_trg)
						fake_img = self.tensor2img(fake_img)
						torchvision.utils.save_image(np.reshape(x_trg.data.cpu(),(-1,1,self.image_size,self.image_size)),
							os.path.join(self.sample_path,
										'real_images-%d_train.png' %(epoch+1)))

						# save the sampled images
						torchvision.utils.save_image(np.reshape(fake_img.data.cpu(),(-1,1,self.image_size,self.image_size)),
							os.path.join(self.sample_path,
										'fake_samples-%d_train.png' %(epoch+1)))

			del x_trg, x_style, style, char, x_trg_c, x_trg_s
			del fake_img, rec_img, id_img, ssim_fake
			#==================================== Valid GD ====================================#
			if (epoch+1) % self.val_step == 0:
				self.generator.train(False)
				self.discriminator.train(False)
				self.generator.eval()
				self.discriminator.eval()

				ssim_fake = 0.
				ssim_rec = 0.
				for i, (x_style, x_trg, _, x_style_c, char, _) in enumerate(self.valid_loader):

					batch_size = x_style.size(0)

					x_style = self.to_variable(x_style)
					x_trg = self.to_variable(x_trg)
					#_, x_style_c = self.enc_char(x_style)
					#_, x_style_c = torch.max(x_style_c, dim=1)
					x_style_c = self.label2onehot(x_style_c)
					char = self.label2onehot(char)

					# Generate fake image from Encoder
					real_style, _  = self.enc_style(x_style)
					real_char, _ = self.enc_char(x_style)
					fake_img = self.generator(x_style_c, real_style, real_char, char)

					ssim_fake += utils.ssim(x_trg, fake_img).data.cpu().numpy()

					rec_img = self.generator(x_style_c, real_style, real_char, x_style_c)

					ssim_rec += utils.ssim(x_style, rec_img).data.cpu().numpy()

				print('Valid SSIM: ', end='')
				print("{:.4f}".format(ssim_fake/(i+1)))
				print('Reocn SSIM: ', end='')
				print("{:.4f}".format(ssim_rec/(i+1)))

				x_trg = x_trg.view(x_trg.size(0), 2, self.image_size, self.image_size)
				x_trg = self.tensor2img(x_trg)
				fake_img = self.tensor2img(fake_img)

				# Save real images
				torchvision.utils.save_image(np.reshape(x_trg.data.cpu(),(-1,1,self.image_size,self.image_size)),
					os.path.join(self.sample_path,
								'real_images-%d_test.png' %(epoch+1)))
				# save the sampled images
				torchvision.utils.save_image(np.reshape(fake_img.data.cpu(),(-1,1,self.image_size,self.image_size)),
					os.path.join(self.sample_path,
								 'fake_samples-%d_test.png' %(epoch+1)))


				# Save the model parameters for each epoch
				g_path = os.path.join(self.model_path, 'generator-%d.pkl' %(epoch+1))
				d_path = os.path.join(self.model_path, 'discriminator-%d.pkl' %(epoch+1))
				se_path = os.path.join(self.model_path, 'style_encoder-%d.pkl' %(epoch+1))
				ce_path = os.path.join(self.model_path, 'char_encoder-%d.pkl' %(epoch+1))
				torch.save(self.generator.state_dict(), g_path)
				torch.save(self.discriminator.state_dict(), d_path)
				torch.save(self.enc_style.state_dict(), se_path)
				torch.save(self.enc_char.state_dict(), ce_path)

			# Decay learning rate
			if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
				g_lr -= (self.g_lr / float(self.num_epochs_decay))
				d_lr -= (self.d_lr / float(self.num_epochs_decay))
				self.update_lr(g_lr, d_lr)
				print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


	#================================ Sampling ====================================/
	#==============================================================================/
	def sample(self):

		"""Translate images using StarGAN trained on a single dataset."""
		se_path = os.path.join(self.model_path, 'style_encoder-%d.pkl' %(self.sample_epochs))
		ce_path = os.path.join(self.model_path, 'char_encoder-%d.pkl' %(self.sample_epochs))

		# Style Encoder Train First
		if os.path.isfile(se_path) and os.path.isfile(ce_path):
			# Load the pretrained Encoder
			self.enc_style.load_state_dict(torch.load(se_path))
			self.enc_char.load_state_dict(torch.load(ce_path))
			print('Enocder is Successfully Loaded from %s'%se_path)

		# Load the trained generator.
		self.restore_model(self.sample_epochs)

		# Set data loader.
		data_loader = self.test_loader

		ssim_fake_total = 0.
		ssim_reco_total = 0.
		with torch.no_grad():
			for i, (x_style, x_trg,  _, x_style_c, char_trg, _) in enumerate(self.test_loader):

				batch_size = x_style.size(0)

				# Prepare input images and target domain labels.
				x_style = x_style.to(self.device)
				x_trg = x_trg.to(self.device)
				#_, x_style_c = self.enc_char(x_style)
				#_, x_style_c = torch.max(x_style_c, dim=1)
				x_style_c = self.label2onehot(x_style_c)
				char_trg = self.label2onehot(char_trg)

				real_style, _ = self.enc_style(x_style)
				real_char, _ = self.enc_char(x_style)

				# Translate images.
				x_fake_list = [x_style, x_trg]
				x_fake = self.generator(x_style_c, real_style, real_char, char_trg)
				x_fake_list.append(x_fake)

				ssim_fake = utils.ssim(x_trg, x_fake).data.cpu().numpy()
				ssim_fake_total += ssim_fake

				print('Valid SSIM: ', end='')
				print("{:.4f}".format(ssim_fake))

				# Translate images.
				x_fake = self.generator(x_style_c, real_style, real_char, x_style_c)
				x_fake_list.append(x_fake)

				ssim_fake = utils.ssim(x_style, x_fake).data.cpu().numpy()
				ssim_reco_total += ssim_fake

				print('Recon SSIM: ', end='')
				print("{:.4f}".format(ssim_fake))

				# Save the translated images.
				x_concat = torch.cat(x_fake_list, dim=2)
				x_concat = self.tensor2img(x_concat)
				result_path = os.path.join(self.result_path, '{}-images.jpg'.format(i+1))
				torchvision.utils.save_image(np.reshape(x_concat.data.cpu(), (-1,1,self.image_size*4,self.image_size)),
					result_path, nrow=batch_size, padding=0)
			print('Saved real and fake images into {}...'.format(result_path))
			print('Total Valid SSIM: ', end='')
			print("{:.4f}".format(ssim_fake_total/(i+1)))
			print('Total Recon SSIM: ', end='')
			print("{:.4f}".format(ssim_reco_total/(i+1)))
