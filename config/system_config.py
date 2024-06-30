from easydict import EasyDict as edict
import time

sys_cfg = edict()

"""default setting"""
sys_cfg.seed = 12345

# model save log dir
sys_cfg.train_id = time.strftime("%m%d%H%M", time.localtime())
sys_cfg.exp_root = './exp_2023'

# training setting
sys_cfg.num_workers = 8

# print interval for training
sys_cfg.log_freq = 10

# evaluation interval for training
sys_cfg.total_epoch = 500
sys_cfg.val_start = 50
sys_cfg.val_freq = 5

# pre-trained weights
sys_cfg.load_imagenet_weight = True

# lambda for supervised loss and unsupervised loss
# sys_cfg.lambda_unsup = 0.5
# sys_cfg.lambda_sup = 1.0

# sys_cfg.exp_name = now + '_' + sys_cfg.dataset + '_' + sys_cfg.model_name
