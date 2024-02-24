import torch
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from io import BytesIO
from IPython import embed
from tqdm import tqdm
import torchvision.transforms as transforms
from pathlib import Path
from random import choice
import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import cv2


def data_augment(img):
    img = np.array(img)

    if random.random() < 0.1:
        sig = sample_continuous([0., 3.])
        gaussian_blur(img, sig)

    if random.random() < 0.1:
        method = sample_discrete(['cv2', 'pil'])
        qual = sample_discrete([30, 100])
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


class Augment():
    def __call__(self, img):
        return data_augment(img)


class DiffusionDataset(Dataset):
    def __init__(self, split, type="LDM", shuf=True, fake_only=False):
        if fake_only:
            root = "deepfake_datasets/diffusion/bedroom/{}/{}/1_fake".format(split, type)
        else:
            root = "deepfake_datasets/diffusion/bedroom/{}/{}".format(split, type)

        self.paths = [(str(x.resolve()), 0 if "real" in str(x.resolve()) else 1) for x in
                      tqdm(list(Path(root).rglob('*.png')))] + \
                     [(str(x.resolve()), 0 if "real" in str(x.resolve()) else 1) for x in
                      tqdm(list(Path(root).rglob('*.jpg')))]

        random.shuffle(self.paths)

        if split == "train":
            self.xform = transforms.Compose([
                transforms.Resize(256),
                Augment(),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
        else:
            self.xform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx]
        img_pil = Image.open(path).convert('RGB')
        img = self.xform(img_pil)
        return img, label
