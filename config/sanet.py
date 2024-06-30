from easydict import EasyDict as edict
from sys_config_back import sys_cfg

cfg = edict()

###########################################
cfg.dataset = sys_cfg.dataset
cfg.dataset_root = sys_cfg.dataset_root
cfg.data_mode = 'fixed_sigma_15'
cfg.crop_size = (400, 400)
cfg.mean_std = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])
cfg.gamma = (0.5, 1.5)
cfg.mst_size = (0.8, 1.2)
cfg.val_size = (1.0, 1.0)

cfg.img_div = 16
cfg.dw = 1
cfg.train_bs = sys_cfg.batch_size
cfg.val_bs = 1
cfg.use_flip = True
cfg.use_mst = sys_cfg.use_mst
cfg.mask = False
cfg.num_workers = 4
