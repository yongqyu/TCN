from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import tqdm


class EncoderFolder(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.images = list(map(lambda x: os.path.join(image_dir+'train', x), os.listdir(image_dir+'train')))
        self.transform = transform
        self.mode = mode

        train_path = 'pretrain_ch.pkl'
        if os.path.isfile(train_path):
            self.train_dataset = torch.load(train_path)
        else:
            self.train_dataset = []
            self.preprocess()
            torch.save(self.train_dataset, train_path)

        self.num_images = len(self.train_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""

        random.seed(1234)
        random.shuffle(self.images)
        for i, img in enumerate(tqdm.tqdm(self.images)):
            style_idx = int(img.split('_')[0][len(self.image_dir+'train/'):])
            char_idx = int(img.split('_')[1][:-len(".png")])

            style_trg_idx = []
            char_trg_idx = []
            style_target = random.choice([x for x in self.images
                                     if str(style_idx)+'_' not in x and '_'+str(char_idx) in x])
            style_trg_idx.append(int(style_target.split('_')[0][len(self.image_dir):]))
            char_target = random.choice([x for x in self.images
                                     if str(style_idx)+'_' in x and '_'+str(char_idx) not in x])
            char_trg_idx.append(int(char_target.split('_')[1][:-len(".png")]))

            self.train_dataset.append([img, style_target, char_target, style_idx, char_idx, style_trg_idx, char_trg_idx])

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        src, style_trg, char_trg, src_style, src_char, trg_style, trg_char = dataset[index]
        src = self.transform(Image.open(src))
        style_trg = self.transform(Image.open(style_trg))
        char_trg = self.transform(Image.open(char_trg))

        return src, style_trg, char_trg, src_style, src_char,\
               torch.LongTensor(trg_style), torch.LongTensor(trg_char)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

class ImageFolder(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.train_images = list(map(lambda x: os.path.join(image_dir+'train', x), os.listdir(image_dir+'train')))
        self.test_images = list(map(lambda x: os.path.join(image_dir+'test', x), os.listdir(image_dir+'test')))
        self.transform = transform
        self.mode = mode

        test_path = 'test_eng.pkl'
        if os.path.isfile(test_path):
            self.test_dataset = torch.load(test_path)
        else:
            self.test_dataset = []
            self.preprocess()
            torch.save(self.test_dataset, test_path)

        if mode == 'train':
            self.num_images = len(self.train_images)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""

        random.seed(1234)
        random.shuffle(self.test_images)
        for i, img in enumerate(self.test_images):
            style_idx = int(img.split('_')[0][len(self.image_dir+'test/'):])
            char_idx = int(img.split('_')[1][:-len(".png")])

            target = random.choice([x for x in self.test_images
                                     if str(style_idx)+'_' in x and '_'+str(char_idx) not in x])
            char_trg_idx = int(target.split('_')[1][:-len(".png")])

            self.test_dataset.append([img, style_idx, char_idx, target, char_trg_idx])

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        if self.mode == 'train':
            random.seed()
            random.shuffle(self.train_images)

            src = self.train_images[index]
            src_style = int(src.split('_')[0][len(self.image_dir+'train/'):])
            src_char = int(src.split('_')[1][:-len(".png")])

            try:
                trg = random.choice([x for x in self.train_images
                                        if str(src_style)+'_' in x and '_'+str(src_char) not in x])
            except:
                trg = src
            trg_char = int(trg.split('_')[1][:-len(".png")])
        else:
            src, src_style, src_char, trg, trg_char = self.test_dataset[index]

        src = self.transform(Image.open(src))
        trg = self.transform(Image.open(trg))

        return src, src_style, src_char, \
               trg, trg_char

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, image_size=128,
               batch_size=16, mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    #transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = ImageFolder(image_dir, transform, mode)
    #dataset = EncoderFolder(image_dir, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
