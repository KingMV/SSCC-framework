from easydict import EasyDict as edict
from .sys_config_back import sys_cfg

cfg = edict()

###########################################
cfg.dataset = sys_cfg.dataset
cfg.dataset_root = sys_cfg.dataset_root
cfg.data_mode = 'fixed_sigma_15'
# cfg.crop_size = (128, 128)
# cfg.crop_size = (576, 768)
cfg.crop_size = (256, 256)
# if sys_cfg.dataset =

cfg.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# cfg.mean_std = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])
if sys_cfg.dataset == 'NWPU':
    cfg.mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
cfg.val_size = (1.0, 1.0)
cfg.ratio = 1.5
cfg.loss_mode = sys_cfg.loss_mode
cfg.img_div = 16
cfg.dw = 8
cfg.train_bs = sys_cfg.batch_size
cfg.val_bs = 1
cfg.num_workers = 4
cfg.fp_dataset = sys_cfg.fpd
# cfg.fp_dataset = 0.1


############################
# data augmentation
# default for unlabel data
cfg.rand_crop = True
cfg.rand_hflip = True
if sys_cfg.dataset == 'NWPU':
    cfg.rand_scale = False
else:
    cfg.rand_scale = True
if cfg.rand_scale:
    cfg.mst_size = (0.8, 1.2)
else:
    cfg.mst_size = (1.0, 1.0)

if sys_cfg.SSM == 'MeanTeacher': #or sys_cfg.SSM == 'NSP':
    cfg.rand_aug = True
else:
    cfg.rand_aug = True
# cfg.add_color_noise = False
# cfg.gamma = (0.5, 1.5)
