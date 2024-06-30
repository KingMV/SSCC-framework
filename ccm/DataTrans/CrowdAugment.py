# some of this code are from: https://github.com/ildoonet/pytorch-randaugment
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import random
import numpy as np

fillcolor = (0, 0, 0)
fillmask = 0
PARAMETER_MAX = 10


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


# Geometric Transformations
def affine_transform(pair, affine_params):
    """

    :param pair: img, den_mask
    :param affine_params:
    :return:
    """
    img, den_mask = pair
    img = img.transform(img.size, Image.AFFINE, affine_params,
                        resample=Image.BILINEAR, fillcolor=fillcolor)
    den_mask = den_mask.transform(den_mask.size, Image.AFFINE, affine_params,
                                  resample=Image.NEAREST, fillcolor=fillmask)
    return img, den_mask


def ShearX(pair, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return affine_transform(pair, (1, v, 0, 0, 1, 0))


def ShearY(pair, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return affine_transform(pair, (1, 0, 0, v, 1, 0))


def TranslateX(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    img, _ = pair
    v = v * img.size[0]
    return affine_transform(pair, (1, 0, v, 0, 1, 0))


def TranslateY(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    img, _ = pair
    v = v * img.size[1]
    return affine_transform(pair, (1, 0, 0, 0, 1, v))


def TranslateXAbs(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return affine_transform(pair, (1, 0, v, 0, 1, 0))


def TranslateYAbs(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return affine_transform(pair, (1, 0, 0, 0, 1, v))


def Rotate(pair, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    img, mask = pair
    img = img.rotate(v, fillcolor=fillcolor)
    mask = mask.rotate(v, resample=Image.NEAREST, fillcolor=fillmask)
    return img, mask


# texture-like transformations

def Solarize(pair, v):  # [0, 256]
    img, mask = pair
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v), mask


def Posterize(pair, v):  # [4, 8]
    img, mask = pair
    assert 4 <= v <= 8
    v = int(v)
    return ImageOps.posterize(img, v), mask


def Grayscale(pair, _):  # [0, 1]
    img, mask = pair
    return ImageOps.grayscale(img).convert('RGB'), mask


def Invert(pair, _):
    img, mask = pair
    return ImageOps.invert(img), mask


def Contrast(pair, v):  # [0.1,1.9]
    img, mask = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Contrast(img).enhance(v), mask


def Color(pair, v):  # [0.1,1.9]
    img, mask = pair
    assert 0.1 <= v <= 1.9
    # ImageEnhance.Hue(img).enhance(v)
    return ImageEnhance.Color(img).enhance(v), mask


def Brightness(pair, v):  # [0.1,1.9]
    img, mask = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Brightness(img).enhance(v), mask


def Sharpness(pair, v):  # [0.1,1.9]
    img, mask = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Sharpness(img).enhance(v), mask


def AutoContrast(pair, _):
    img, mask = pair
    return ImageOps.autocontrast(img), mask


def Equalize(pair, _):
    img, mask = pair
    return ImageOps.equalize(img), mask


def Flip(pair, _):  # not from the paper
    img, mask = pair
    return ImageOps.mirror(img), ImageOps.mirror(mask)


def Cutout(pair, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return pair
    img, mask = pair
    v = v * img.size[0]
    return CutoutAbs(img, v), mask


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
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
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Identity(pair, v):
    return pair


def texture_augment_pool():
    augs = [(AutoContrast, 0, 1),
            # (Invert, 0, 1),
            (Equalize, 0, 1),
            (Solarize, 0, 110),
            (Posterize, 4, 8),
            (Color, 0.1, 1.9),
            (Brightness, 0.1, 1.9),
            (Sharpness, 0.1, 1.9),
            (Grayscale, 0, 1),
            ]
    return augs


def geometric_augment_pool():
    augs = [(ShearX, 0., 0.3),
            (ShearY, 0., 0.3),
            (TranslateX, 0., 0.45),
            (TranslateY, 0., 0.45),
            (Rotate, 0, 30),
            (Identity, 0., 1.),
            (Flip, 1, 1)
            ]
    return augs


class RandTextureAugment_vis(object):
    """
    Randomly select several texture augmentations from the pool and apply it to the image.
    """

    def __init__(self, N, M):
        self.N = N  # number of augmentations, it determines the strength of augmentation
        self.M = M
        self.augment_pool = texture_augment_pool()

    def __call__(self, img, mask, aug_id):
        pair = img, mask
        # ops = random.choices(self.augment_pool, k=self.N)
        # ops = random.choices(self.augment_pool, k=self.N)
        ops = [self.augment_pool[aug_id]]
        # print(ops[0][0].__name__)
        ops_name = ops[0][0].__name__
        # v = np.random.randint(1, self.M)
        v = self.M
        for op, min_val, max_val in ops:
            val = (float(v) / 30) * float(max_val - min_val) + min_val
            # print(val)
            # print(op.__name__, val)
            pair = op(pair, val)
        return pair[0], [ops_name, val]

        # if random.random() < 0.5:
        #     op, min_val, max_val = random.choice(self.pool)
        #     val = random.uniform(min_val, max_val)
        #     return op(pair, val)
        # else:
        #     return pair


class RandTextureAugment(object):
    """
    Randomly select several texture augmentations from the pool and apply it to the image.
    """

    def __init__(self, N, M):
        self.N = N  # number of augmentations, it determines the strength of augmentation
        self.M = M
        self.augment_pool = texture_augment_pool()

    def __call__(self, img, mask):
        pair = img, mask
        ops = random.choices(self.augment_pool, k=self.N)
        # print(ops[0][0].__name__)
        # ops_name = ops
        for op, min_val, max_val in ops:
            val = (float(self.M) / 30) * float(max_val - min_val) + min_val
            # print(op.__name__, val)
            pair = op(pair, val)
        return pair

        # if random.random() < 0.5:
        #     op, min_val, max_val = random.choice(self.pool)
        #     val = random.uniform(min_val, max_val)
        #     return op(pair, val)
        # else:
        #     return pair


class RandGeometricAugment(object):
    """
    Randomly select several geometric augmentations from the pool and apply it to the image.
    """

    def __init__(self, N=3, M=10):
        self.N = N  # number of augmentations, it determines the strength of augmentation
        self.M = M
        self.augment_pool = geometric_augment_pool()

    def __call__(self, img, mask):
        pair = img, mask
        ops = random.choices(self.augment_pool, k=self.N)

        for op, min_val, max_val in ops:
            val = (float(self.M) / 30) * float(max_val - min_val) + min_val
            print(op.__name__, val)
            pair = op(pair, val)
        return pair
