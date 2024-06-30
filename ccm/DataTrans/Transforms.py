from PIL import Image
import numpy as np
import cv2
import random
import torch
import torchvision.transforms.functional as F
import numbers
from PIL import ImageFilter
import torchvision.transforms as tf


def add_gaussian_noise(image, mean=0, var=0.01):
    if Image.isImageType(image):
        image = np.asarray(image)
    image = np.asarray(image / 255, dtype=np.float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    image = image + noise
    image = np.clip(image, 0, 1.0)
    image = np.asarray(image * 255, dtype=np.uint8)
    image = Image.fromarray(image)

    return image


def _pad_(x1, x2, min_size, p_mode='constant', fill=0):
    w, h = x1.size
    if w < min_size[0] or h < min_size[1]:
        pad_w = min_size[0] - w
        pad_h = min_size[1] - h
        pad_l = max(0, pad_w // 2)
        pad_r = max(0, pad_w - pad_l)
        pad_t = max(0, pad_h // 2)
        pad_b = max(0, pad_h - pad_t)
        x1 = F.pad(x1, (pad_l, pad_t, pad_r, pad_b), fill, p_mode)
        if x2 is not None:
            x2 = np.pad(x2, ((pad_t, pad_b), (pad_l, pad_r)), mode=p_mode, constant_values=0)
    return x1, x2


def _rand_gray_scale(img, p=0.2):
    assert isinstance(img, Image.Image), 'the input img should be Image'
    return tf.RandomGrayscale(p=p)(img)


def _rand_color_jitter(img, p=0.5):
    assert isinstance(img, Image.Image), 'the input img should be Image'
    jitter_fn = tf.ColorJitter(brightness=[0.5, 1.5],  # 1 is not change
                               contrast=[0.5, 1.5],  # 1 is not change
                               saturation=[0.5, 1.5],  # 1 is not change
                               hue=[-0.5, 0.5])  # 0 is not change
    if random.random() < p:
        img = jitter_fn(img)
    return img


def _gaussian_blur(img, sigma=[0.1, 0.2], p=0.2):
    assert isinstance(img, Image.Image), 'the input img should be Image'
    s = random.uniform(sigma[0], sigma[1])
    if random.random() < p:
        img = img.filter(ImageFilter.GaussianBlur(radius=s))
    return img


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        '''

        :param img:  PIL
        :param mask: numpy array
        :return:
        '''
        w, h = img.size
        th, tw = self.size

        if w == tw and h == th:
            return img, mask

        img, mask = _pad_(img, mask, [tw, th])
        w, h = img.size
        # if w < tw or h < th:
        #     return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask[y1:y1 + th, x1:x1 + tw]


class RandomMultiScale(object):
    def __init__(self, scale):
        self.scale_factor = scale

    def __call__(self, img, mask):
        if img is not None:
            assert isinstance(img, Image.Image), 'the input img should be Image'
        if isinstance(self.scale_factor, list):
            scale_list = self.scale_factor
        elif isinstance(self.scale_factor, numbers.Number):
            scale_list = (self.scale_factor, self.scale_factor)
        else:
            raise ValueError('the scale object should be list')

        if scale_list[0] == 1 and scale_list[1] == 1:
            return img, mask

        temp_aspect_ratio = 1.0
        temp_scale = scale_list[0] + (scale_list[1] - scale_list[0]) * random.random()

        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio

        scaled_size = (int(img.size[0] * scale_factor_x),
                       int(img.size[1] * scale_factor_y))  # width, height

        if img is not None:
            img = img.resize(scaled_size, Image.BILINEAR)
        if mask is not None:
            factor = (scaled_size[0] / mask.shape[1]) * (scaled_size[1] / mask.shape[0])
            mask = cv2.resize(mask, scaled_size, interpolation=cv2.INTER_LINEAR) / factor

        return img, mask


class RandomHFlip(object):
    def __init__(self, p):
        self.prob = p

    def __call__(self, img, mask):
        if random.random() < self.prob:  # 随机生成0-1之间的浮点数 ，每次执行生成的不一样
            return img.transpose(Image.FLIP_LEFT_RIGHT), np.fliplr(mask)
        return img, mask


class BaseTransform(object):
    def __init__(self, args):
        self.d_ratio = args.DEN_DIV
        self.mean_std = args.MEAN_STD

        self.base_trans = Compose([RandomMultiScale(scale=args.MST_FACTOR),
                                   RandomCrop(size=args.CROP_SIZE),
                                   RandomHFlip(p=0.5),
                                   ])

    def __call__(self, data_blob):
        assert isinstance(data_blob, dict)

        name = data_blob['name']
        img = data_blob['image']
        den_map = data_blob['den']
        is_label = data_blob['is_label']
        aug_img_tensor = None
        img, den_map = self.base_trans(img, den_map)

        aug_img = img.copy()
        aug_img = _rand_color_jitter(aug_img, p=0.8)
        aug_img = _rand_gray_scale(aug_img, p=0.2)
        aug_img = _gaussian_blur(aug_img, sigma=[0.1, 0.2], p=0.2)

        # downsample density map to the same size as the output of model
        down_w = img.size[0] // self.d_ratio
        down_h = img.size[1] // self.d_ratio
        factor = (down_w / img.size[0]) * (down_h / img.size[1])
        den_map = cv2.resize(den_map, (down_w, down_h), interpolation=cv2.INTER_LINEAR) / factor
        den_map = den_map[np.newaxis, :, :]

        # to tensor
        img_tensor = F.to_tensor(img)
        aug_img_tensor = F.to_tensor(aug_img)
        # normalize image
        img_tensor = F.normalize(img_tensor, *self.mean_std)
        aug_img_tensor = F.normalize(aug_img_tensor, *self.mean_std)

        db_new = {'name': name,
                  'img': img_tensor,
                  'den': torch.from_numpy(den_map.copy()).float(),
                  'is_label': is_label,
                  'aug_img': aug_img_tensor if aug_img_tensor is not None else img_tensor,
                  }

        return db_new


class TestTransform(object):
    def __init__(self, args):
        self.d_ratio = args.DEN_DIV
        self.mean_std = args.MEAN_STD

    def __call__(self, db):
        assert isinstance(db, dict)
        name = db['name']
        img = db['image']
        den_map = db['den']

        # downsample density map
        down_w = img.size[0] // self.d_ratio
        down_h = img.size[1] // self.d_ratio
        factor = (down_w / img.size[0]) * (down_h / img.size[1])

        # to tensor
        img_tensor = F.to_tensor(img)
        img_new_tensor = img_tensor

        # normalize image
        img_new_tensor = F.normalize(img_new_tensor, *self.mean_std)

        den = cv2.resize(den_map, (down_w, down_h), interpolation=cv2.INTER_LINEAR) / factor

        db_new = {'name': name,
                  'img': img_new_tensor,
                  'den': torch.from_numpy(den.copy()).float()
                  }

        return db_new
