import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def main(config):
    cudnn.benchmark = True

    train_loader = get_loader(image_path=config.train_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers)
    valid_loader = get_loader(image_path=config.valid_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers)
    test_loader = get_loader(image_path=config.test_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers)

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    # Train and sample the images
    if config.mode == 'train':
        solver = Solver(config, train_loader, valid_loader, test_loader)
        solver.train()
    elif config.mode == 'sample':
        solver = Solver(config, None, None, test_loader)
        solver.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--d_train_repeat', type=int, default=5)
    parser.add_argument('--enc2mat', type=bool, default=False)

    # training hyper-parameters
    parser.add_argument('--enc_epochs', type=int, default=100)
    parser.add_argument('--sample_epochs', type=int, default=100)
    parser.add_argument('--start_epochs', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_epochs_decay', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--g_lr', type=float, default=0.0002)
    parser.add_argument('--d_lr', type=float, default=0.0002)
    parser.add_argument('--e_lr', type=float, default=0.0002)
    parser.add_argument('--lambda_cls', type=float, default=1)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--train_path', type=str, default='../dataset/small-trans-valid/train/')
    parser.add_argument('--valid_path', type=str, default='../dataset/small-trans-valid/valid/')
    parser.add_argument('--test_path', type=str, default='../dataset/small-trans-valid/test/')
    parser.add_argument('--log_step', type=int , default=3000)
    parser.add_argument('--val_step', type=int , default=5)
    parser.add_argument('--style_cnt', type=int , default=150)
    parser.add_argument('--char_cnt', type=int , default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
