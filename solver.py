from model import EncoderList, Generator
from model import Discriminator
from model import ClassifierList
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import utils


class Solver(object):
    """Solver for training and testing TCN."""

    def __init__(self, data_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.data_loader = data_loader

        # Model configurations.
        self.image_size = config.image_size
        self.style_cnt = config.style_cnt
        self.char_cnt = config.char_cnt
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_ssim= config.lambda_ssim
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.batch_size = config.batch_size
        self.enc_iters = config.enc_iters
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.E = EncoderList(self.image_size, self.g_conv_dim, self.style_cnt, self.char_cnt)
        self.G = Generator(self.g_conv_dim, self.char_cnt, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.style_cnt, self.char_cnt, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(list(self.E.parameters()) + list(self.G.parameters()),
                                            self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.E, 'E')
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.E.to(self.device)
        self.G.to(self.device)
        self.D.to(self.device)

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
        print('Loading the trained models from step {}...'.format(resume_iters))
        E_path = os.path.join(self.model_save_dir, '{}-E.ckpt'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.E.load_state_dict(torch.load(E_path, map_location=lambda storage, loc: storage))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def restore_cls_model(self):
        """Restore the trained generator and discriminator."""
        self.C = ClassifierList(self.image_size, self.g_conv_dim, self.style_cnt, self.char_cnt).to(self.device)
        C_path = os.path.join(self.model_save_dir, 'C.ckpt')
        self.C.load_state_dict(torch.load(C_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

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

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out.to(self.device)

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def classification_acc(self, logit, target):
        """Compute accuracy."""
        logit = torch.max(logit, 1)[1]
        target = target
        correct_prediction = (logit == target).float()
        return torch.mean(correct_prediction)

    def pretrain(self):
        """Pretrain TCN."""

        # data loader slit
        total_len = len(self.data_loader)
        train_thr = int(total_len*0.8)
        data_iter = iter(self.data_loader)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        loss = {'E/acc_style':0, 'E/acc_char':0}
        acc_style = 0.
        acc_char = 0.

        # Start training.
        print('Start pre-training...')
        for i in range(self.enc_iters):
            if (i % total_len) < train_thr:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Fetch real images and labels.
                try:
                    x_real, style_trg, char_trg, x_style, x_char, trg_style, trg_char = next(data_iter)
                except:
                    log = "Iteration [{}/{}]".format(i, self.num_iters)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value/(total_len-train_thr))
                    print(log)
                    acc_style = loss['E/acc_style']/(total_len-train_thr)
                    acc_char  = loss['E/acc_char']/(total_len-train_thr)
                    loss = {'E/acc_style':0, 'E/acc_char':0}

                    data_iter = iter(self.data_loader)
                    x_real, style_trg, char_trg, x_style, x_char, trg_style, trg_char = next(data_iter)

                batch_size = x_real.size(0)
                # Generate real labels
                x_real = x_real.to(self.device)
                x_style= x_style.to(self.device)
                x_char = x_char.to(self.device)

                style_trg = style_trg.to(self.device)
                trg_style = trg_style.to(self.device)
                char_trg = char_trg.to(self.device)
                trg_char = trg_char.to(self.device)
                # =================================================================================== #
                #                                   2. Train the encoder                              #
                # =================================================================================== #

                # Compute loss with real images.
                x_sout, x_cout, cls_style, cls_char = self.E(x_real)
                e_loss_style = self.classification_loss(cls_style, x_style)
                e_loss_char  = self.classification_loss(cls_char, x_char)

                style_sout, style_cout, _, _ = self.E(style_trg)
                char_sout, char_cout, _, _ = self.E(char_trg)

                triplet_loss_style = F.triplet_margin_loss(x_sout, char_sout, style_sout, margin=1)
                triplet_loss_char  = F.triplet_margin_loss(x_cout, style_cout, char_cout, margin=1)

                # Backward and optimize.
                e_loss = e_loss_style + e_loss_char
                if acc_style > 0.8:
                    e_loss += triplet_loss_style
                if acc_char  > 0.8:
                    e_loss += triplet_loss_char

                self.reset_grad()
                e_loss.backward()
                self.g_optimizer.step()

            else:
                with torch.no_grad():
                    x_real, _, _, x_style, x_char, _, _ = next(data_iter)

                    # Prepare input images and target domain labels.
                    x_real = x_real.to(self.device)
                    x_style= x_style.to(self.device)
                    x_char = x_char.to(self.device)

                    # Compute loss with real images.
                    _, _, out_style, out_char = self.E(x_real)
                    d_acc_style= self.classification_acc(out_style, x_style)
                    d_acc_char = self.classification_acc(out_char, x_char)

                    # Logging.
                    loss['E/acc_style']+= d_acc_style.item()
                    loss['E/acc_char'] += d_acc_char.item()

            # Save model checkpoints.
            if (i+1) % 3000 == 0:
                E_path = os.path.join(self.model_save_dir, '{}-E.ckpt'.format(i+1))
                torch.save(self.E.state_dict(), E_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                for param_group in self.g_optimizer.param_groups:
                    param_group['lr'] = g_lr
                print ('Decayed learning rates, e_lr: {}.'.format(g_lr))


    def train(self):
        """Train TCN."""
        # Start training from scratch or resume training.
        start_iters = 0
        E_path = os.path.join(self.model_save_dir, 'E.ckpt'.format(self.enc_iters))
        if not os.path.isfile(E_path):
            self.pretrain()
        pretrained_E_dict = torch.load(E_path, map_location=lambda storage, loc: storage)
        E_dict = self.E.state_dict()
        pretrained_E_dict = {k: v for k, v in pretrained_E_dict.items() if k in E_dict}
        E_dict.update(pretrained_E_dict)
        self.E.load_state_dict(E_dict)
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Fetch fixed inputs for debugging.
        data_iter = iter(self.data_loader)
        x_fixed, x_fixed_style, x_fixed_char, y_fixed, y_fixed_char = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        y_fixed = y_fixed.to(self.device)
        y_fixed_char = y_fixed_char.to(self.device)
        c_fixed_list = [(self.label2onehot(x_fixed_char, self.char_cnt),
                         self.label2onehot(y_fixed_char, self.char_cnt))]

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, x_style, x_char, y_trg, y_char = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                x_real, x_style, x_char, y_trg, y_char = next(data_iter)

            batch_size = x_real.size(0)
            # Generate real labels
            x_real = x_real.to(self.device)
            x_style= x_style.to(self.device)
            x_char = x_char.to(self.device)
            x_char_onehot = self.label2onehot(x_char, self.char_cnt)
            # Character transfer. keep style. and thats' character index

            y_trg  = y_trg.to(self.device)
            y_char = y_char.to(self.device)
            y_char_onehot = self.label2onehot(y_char, self.char_cnt)
            # Style transfer. keep character. and thats' style index

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_style, out_char = self.D(y_trg)
            d_loss_real  = torch.mean((out_src - 1) ** 2)
            d_loss_style = self.classification_loss(out_style, x_style)
            d_loss_char  = self.classification_loss(out_char, y_char)
            d_acc_char  = self.classification_acc(out_char, y_char)

            # Compute loss with fake images.
            style_enc, char_enc, _, _  = self.E(x_real)
            y_fake = self.G(x_char_onehot, style_enc, char_enc, y_char_onehot)
            fake_src, _, _ = self.D(y_fake.detach())
            d_loss_fake = torch.mean(fake_src ** 2)

            # Compute loss for gradient penalty.
            alpha = torch.rand(y_trg.size(0), 1, 1, 1).to(self.device)
            y_hat = (alpha * y_trg.data + (1 - alpha) * y_fake.data).requires_grad_(True)
            gp_src, _, _ = self.D(y_hat)
            d_loss_gp = self.gradient_penalty(gp_src, y_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * (d_loss_style + d_loss_char)\
                                               + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_style'] = d_loss_style.item()
            loss['D/loss_char'] = d_loss_char.item()
            loss['D/acc_char'] = d_acc_char.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                style_enc, char_enc, _, _ = self.E(x_real)
                y_fake = self.G(x_char_onehot, style_enc, char_enc, y_char_onehot)
                out_src, out_style, out_char = self.D(y_fake)

                g_loss_fake  = torch.mean((out_src - 1) ** 2)
                g_loss_style = self.classification_loss(out_style, x_style)
                g_loss_char  = self.classification_loss(out_char, y_char)
                g_acc_style = self.classification_acc(out_style, x_style)
                g_acc_char  = self.classification_acc(out_char, y_char)

                # Training G to 'y_fake' and 'y_trg' are similar. L1 loss
                g_loss_l1 = torch.mean(torch.abs(y_trg - y_fake))

                # Compute Structural similarity measure of the Generator
                g_loss_ssim = utils.ssim(y_trg, y_fake)

                # Target-to-original domain.
                style_fenc, char_fenc, _, _  = self.E(y_fake)
                x_reconst = self.G(y_char_onehot, style_fenc, char_fenc, x_char_onehot)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Reconstruct Perceptual Loss
                style_renc, char_renc, _, _ = self.E(x_reconst)
                g_loss_percept = torch.mean((style_enc - style_renc) ** 2) +\
                                 torch.mean((char_enc - char_renc) ** 2)

                x_fake = self.G(x_char_onehot, style_enc, char_enc, x_char_onehot)
                g_loss_id = torch.mean(torch.abs(x_real - x_fake))

                # Backward and optimize.
                g_loss = g_loss_fake + g_loss_style \
                                     + self.lambda_cls * (g_loss_char) \
                                     + self.lambda_rec * (g_loss_rec + g_loss_percept + g_loss_id)
                                     + self.lambda_ssim* (g_loss_l1 - g_loss_ssim)
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_style'] = g_loss_style.item()
                loss['G/loss_char'] = g_loss_char.item()
                loss['G/acc_char'] = g_acc_char.item()
                loss['G/loss_l1'] = g_loss_l1.item()
                loss['G/loss_ssim'] = g_loss_ssim.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_per'] = g_loss_percept.item()
                loss['G/loss_id'] = g_loss_id.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_real, y_trg, y_fake, x_fixed, y_fixed]
                    style_fixed_enc, char_fixed_enc, _, _ = self.E(x_fixed)
                    for (c_ffixed, c_tfixed) in c_fixed_list:
                        x_fake_list.append(self.G(c_ffixed, style_fixed_enc, char_fixed_enc, c_tfixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(x_concat.data.cpu(), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                E_path = os.path.join(self.model_save_dir, '{}-E.ckpt'.format(i+1))
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.E.state_dict(), E_path)
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def test(self):
        """Translate images using trained TCN."""
        from sklearn.metrics import accuracy_score

        # Load the trained generator.
        self.restore_model(self.test_iters)
        self.restore_cls_model()

        l1_rec = 0.
        ssim_rec = 0.
        l1_test = 0.
        ssim_test = 0.
        style_acc = 0.
        char_acc = 0.
        style_acc_rec = 0.
        char_acc_rec = 0.
        with torch.no_grad():
            for i, (x_real, x_style, x_char, y_trg, y_char) in enumerate(self.data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                y_trg = y_trg.to(self.device)
                x_char = x_char.to(self.device)
                x_char_onehot = self.label2onehot(x_char, self.char_cnt)
                y_char = y_char.to(self.device)
                y_char_onehot = self.label2onehot(y_char, self.char_cnt)

                # Translate images.
                fake_list = [x_real, y_trg]
                style_enc, char_enc, _, _ = self.E(x_real)
                x_fake = self.G(x_char_onehot, style_enc, char_enc, x_char_onehot)
                fake_list.append(x_fake)
                _, _, style_cls_rec, char_cls_rec = self.C(x_fake)
                y_fake = self.G(x_char_onehot, style_enc, char_enc, y_char_onehot)
                fake_list.append(y_fake)
                _, _, style_cls, char_cls = self.C(y_fake)

                loss_l1_rec = torch.mean(torch.abs(x_real - x_fake))
                loss_ssim_rec = utils.ssim(x_real, x_fake)
                loss_l1 = torch.mean(torch.abs(y_trg - y_fake))
                loss_ssim = utils.ssim(y_trg, y_fake)

                acc_style_rec = accuracy_score(x_style.cpu().numpy(),
                                               torch.max(style_cls_rec, 1)[1].cpu().numpy())
                acc_char_rec = accuracy_score(x_char.cpu().numpy(),
                                              torch.max(char_cls_rec, 1)[1].cpu().numpy())
                acc_style = accuracy_score(x_style.cpu().numpy(),
                                           torch.max(style_cls, 1)[1].cpu().numpy())
                acc_char = accuracy_score(y_char.cpu().numpy(),
                                          torch.max(char_cls, 1)[1].cpu().numpy())

                l1_rec += loss_l1_rec.item()
                ssim_rec += loss_ssim_rec.item()
                l1_test += loss_l1.item()
                ssim_test += loss_ssim.item()
                style_acc_rec += acc_style_rec
                char_acc_rec += acc_char_rec
                style_acc += acc_style
                char_acc += acc_char

                # Save the translated images.
                x_concat = torch.cat(fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(x_concat.data.cpu(), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
            print('[Rec L1] : {} [Rec SSIM] : {} [Rec Style Acc] : {} [Rec Char Acc] : {} \
                   [TC L1] : {} [TC SSIM] : {}, [Style Acc] : {} [Char Acc] : {}'.format(
                   l1_rec/(i+1), ssim_rec/(i+1), style_acc_rec/(i+1), char_acc_rec/(i+1),
                    l1_test/(i+1), ssim_test/(i+1), style_acc/(i+1), char_acc/(i+1)))
