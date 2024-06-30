from easydict import EasyDict as edict

SHHA_CFG = edict()

SHHA_CFG.ROOT = '../Dataset'
SHHA_CFG.CROP_SIZE = (400, 400)
SHHA_CFG.MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
SHHA_CFG.DEN_DIV = 8
SHHA_CFG.MST_FACTOR = [0.8, 1.2]
SHHA_CFG.VAL_BS = 1
