# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image


def ShearX(img, v, den=None):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    if den is not None:
        return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0)), \
               den.transform(den.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0)),
    else:
        return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, den=None):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    if den is not None:
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0)), \
               den.transform(den.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))
    else:
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v, den=None):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    if den is not None:
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)), \
               den.transform(den.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
    else:
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v, den=None):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    if den is not None:
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)), \
               den.transform(den.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
    else:
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, den=None):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    if den is not None:
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)), \
               den.transform(den.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
    else:
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v, den=None):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    if den is not None:
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)), \
               den.transform(den.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
    else:
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v, den=None):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    if den is not None:
        return img.rotate(v), den.rotate(v)
    else:
        return img.rotate(v)


# def Rotate(img, v, den=None):  # [-30, 30]
#     assert -30 <= v <= 30
#     if random.random() > 0.5:
#         v = -v
#     if den is not None:
#         return img.rotate(v), den.rotate(v)


def AutoContrast(img, _, den=None):
    if den is not None:
        return PIL.ImageOps.autocontrast(img), den
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _, den=None):
    if den is not None:
        return PIL.ImageOps.invert(img), den
    else:
        return PIL.ImageOps.invert(img)


def Equalize(img, _, den=None):
    if den is not None:
        return PIL.ImageOps.equalize(img), den
    else:
        return PIL.ImageOps.equalize(img)


def Flip(img, _, den=None):  # not from the paper
    if den is not None:
        return PIL.ImageOps.mirror(img), PIL.ImageOps.mirror(den)
    else:
        return PIL.ImageOps.mirror(img)


def Solarize(img, v, den=None):  # [0, 256]
    assert 0 <= v <= 256
    if den is not None:
        return PIL.ImageOps.solarize(img, v), den
    else:
        return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128, den=None):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    if den is not None:
        return PIL.ImageOps.solarize(img, threshold), den
    else:
        return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v, den=None):  # [4, 8]
    v = int(v)
    v = max(1, v)
    if den is not None:
        return PIL.ImageOps.posterize(img, v), den
    else:
        return PIL.ImageOps.posterize(img, v)


def Contrast(img, v, den=None):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    if den is not None:
        return PIL.ImageEnhance.Contrast(img).enhance(v), den
    else:
        return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v, den=None):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    if den is not None:
        return PIL.ImageEnhance.Color(img).enhance(v), den
    else:
        return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v, den=None):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    if den is not None:
        return PIL.ImageEnhance.Brightness(img).enhance(v), den
    else:
        return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v, den=None):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    if den is not None:
        return PIL.ImageEnhance.Sharpness(img).enhance(v), den
    else:
        return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v, den=None):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    # color = (125, 123, 114)
    color = (0, 0, 0)
    img = img.copy()

    if den is not None:
        return PIL.ImageDraw.Draw(img).rectangle(xy, color), PIL.ImageDraw.Draw(den).rectangle(xy, 0)
    else:
        return PIL.ImageDraw.Draw(img).rectangle(xy, color)


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


# def Flip(img, v, den=None):
#     if den is not None:
#         return img.transpose(Image.FLIP_LEFT_RIGHT), den.transpose(Image.FLIP_LEFT_RIGHT)
#     return img.transpose(Image.FLIP_LEFT_RIGHT)


def Identity(img, v, den=None):
    if den is not None:
        return img, den
    return img


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # l = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        # (Flip, 0, 1),
        (Identity, 0, 1),
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        # (Invert, 0, 1),
        # (Rotate, 0, 30),
        # (Posterize, 0, 4),
        # (Solarize, 0, 256),
        # (SolarizeAdd, 0, 110),
        # (Color, 0.5, 1.0),
        (Contrast, 0.5, 1.5),
        (Brightness, 0.5, 1.5),
        (Sharpness, 0.5, 1.5),
        # (ShearX, 0., 0.3),
        # (ShearY, 0., 0.3),
        # (CutoutAbs, 0, 40),
        # (CutoutDefault, 0, 40),
        # (TranslateXabs, 0., 100),
        # (TranslateYabs, 0., 100),
    ]

    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


# class CutoutDefault(object):
#     """
#     Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
#     """
#
#     def __init__(self, length):
#         self.length = length
#
#     def __call__(self, img, v, den=None):
#         h, w = img.size(1), img.size(2)
#         mask = np.ones((h, w), np.float32)
#         y = np.random.randint(h)
#         x = np.random.randint(w)
#
#         y1 = np.clip(y - self.length // 2, 0, h)
#         y2 = np.clip(y + self.length // 2, 0, h)
#         x1 = np.clip(x - self.length // 2, 0, w)
#         x2 = np.clip(x + self.length // 2, 0, w)
#
#         mask[y1: y2, x1: x2] = 0.
#
#         mask = torch.from_numpy(mask)
#         mask = mask.expand_as(img)
#         img *= mask
#         if den is not None:
#             return img, den * mask
#         else:
#             return img


def CutoutDefault(img, v, den=None):
    # h, w = img.size(1), img.size(2)
    w, h = img.size
    mask = np.ones((h, w), np.float32)
    y0 = np.random.randint(h)
    x0 = np.random.randint(w)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))

    # y1 = np.clip(y - v // 2, 0, h)
    # y2 = np.clip(y + v // 2, 0, h)
    # x1 = np.clip(x - v // 2, 0, w)
    # x2 = np.clip(x + v // 2, 0, w)

    mask[y0: y1, x0: x1] = 0.

    # mask = torch.from_numpy(mask)
    if den is not None:
        den = den * mask
        mask = np.expand_dims(mask, axis=-1).repeat(3, axis=-1)
        img *= mask
        return img, den
    else:
        img *= mask
        return img


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img, den=None, att=None):
        ops = random.choices(self.augment_list, k=self.n)
        # ops = self.augment_list[-3:]
        op_names = []
        for op, minval, maxval in ops:
            # val = (float(self.m) / 30) * float(maxval - minval) + minval
            val = minval + float(maxval - minval) * random.random()
            # print(op.__name__)
            op_names.append(op.__name__)
            op_names.append(val)
            if den is not None:
                img, den = op(img, val, den=den)
            else:
                img = op(img, val)
        return img, den, op_names
