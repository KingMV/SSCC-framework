from easydict import EasyDict as edict
import time
import os

sys_cfg = edict()

sys_cfg.model_name = 'Baseline1'
sys_cfg.SSM = 'MeanTeacher'  # MeanTeacher UDA SSKT PseudoLabel  CRDA  SSCC, NSP
# UAS_MEAN  MC_DROP UC_Normal MDN_Normal
sys_cfg.loss_mode = 0  # 0:mse loss for labeled data
# if sys_cfg.model_name == 'Baseline2':
#     sys_cfg.loss_mode = 1  # 0:mse loss for labeled data
# common setting
sys_cfg.batch_size = 24
sys_cfg.batch_label_size = 12
sys_cfg.dataset = 'shha'
sys_cfg.bn_dtype = 'none'  # 是 split
sys_cfg.use_adv_dataset = False
if sys_cfg.SSM == 'MeanTeacher':
    sys_cfg.bn_dtype = 'dsbn'  # split mix

# sys_cfg.perturbation_region = 'foreground'
sys_cfg.perturbation_ways = 'BCG'  # G:grayscale, C:colorjitter, H:
sys_cfg.perturbation_region = 'ALL'  # background ALL
# if 'H' in sys_cfg.perturbation_ways:
#     sys_cfg.perturbation_region = 'ALL'  # background ALL

# if sys_cfg.model_name == 'CSeg':
#     sys_cfg.use_seg_branch = True
# else:
#     sys_cfg.use_seg_branch = False

if sys_cfg.model_name == 'CSeg':
    sys_cfg.use_seg_branch = True
else:
    sys_cfg.use_seg_branch = False
# sys_cfg.input_data_mode = 'only_label'  # 训练数据的监督级别
# [16,16]表示一个batchsize全是有标签


################################################
# Dataset setting
sys_cfg.use_aug_dataset = True

#################################
# KTSS
sys_cfg.stage_inter = 1  # the interval of memory model updating

##########################3

sys_cfg.consistency_rampup = 300
sys_cfg.consistency = 1.0
sys_cfg.ema_decay = 0.9

sys_cfg.fpd = 0.1  # 有标签数据的比例
sys_cfg.add_aug = True

sys_cfg.k_trans_num = 4  # only true for UAS_MEAN strategy

# loss setting
sys_cfg.alpha_1 = 1  # parameter for mse loss
sys_cfg.alpha_2 = 0.9
sys_cfg.cutmix = False
sys_cfg.output_feature = False

if sys_cfg.dataset == 'shha':
    if sys_cfg.loss_mode == 1:
        sys_cfg.dataset_root = '../Dataset/ShanghaiTech_bl'
    else:
        sys_cfg.dataset_root = '../Dataset'
        # sys_cfg.dataset_root = '../Crowd_Dataset_sfanet'
elif sys_cfg.dataset == 'shhb':
    if sys_cfg.loss_mode == 'bl':
        sys_cfg.dataset_root = '../Dataset/ShanghaiTech_bl'
    else:
        sys_cfg.dataset_root = '../Dataset'
elif sys_cfg.dataset == 'qnrf':
    sys_cfg.dataset_root = '../Dataset'
elif sys_cfg.dataset == 'ucf50':
    sys_cfg.dataset_root = '../Dataset'
elif sys_cfg.dataset == 'NWPU':
    sys_cfg.dataset_root = '../Dataset'
elif sys_cfg.dataset == 'JHU':
    sys_cfg.dataset_root = '../Dataset'
# sys_cfg.downsampling = 1

sys_cfg.use_mst = False
sys_cfg.load_weights = True
sys_cfg.freeze_frontend = False
# ADD dropout layer#
sys_cfg.enable_drop = False
sys_cfg.dropout_rate = 0.3

if sys_cfg.SSM == 'MC_DROP' or sys_cfg.SSM == 'MC_DROP_pixel':
    sys_cfg.enable_drop = True

# uncertainty configure parameters
if sys_cfg.SSM == 'UC_Normal' or sys_cfg.SSM == 'UC_MT' or sys_cfg.SSM == 'MDN_Normal':
    # if sys_cfg.SSM == 'UC_Normal' or  sys_cfg.SSM == 'MDN_Normal':
    sys_cfg.use_uncertainty = True
else:
    sys_cfg.use_uncertainty = False

sys_cfg.load_FEN = False
sys_cfg.pre_model_path = './exp_2021/06030259_shha_Baseline1_UC_Normal_0_0.1/checkpoint/' \
                         't_net_ep_71_mae_94.81_mse_164.07_pmae_7.01_pmse_18.34_p_21.27_s_0.69.pth'

sys_cfg.resume = False
sys_cfg.resume_path = './exp_2021/08290610_NWPU_Baseline1_MeanTeacher_0_0.1/checkpoint/latest_state.pth'

sys_cfg.tsg_model_name = 'RankNet'
sys_cfg.load_ss_weight = False
# sys_cfg.model_name = 'CSRNet_USL_RE'
# sys_cfg.model_name = 'CSRNet'

# sys_cfg.model_name = 'Crowd_SPN'
# sys_cfg.model_name = 'RankNet'
# sys_cfg.model_name = 'CSRNet_SSL'
# sys_cfg.model_name = 'MCNN'
# sys_cfg.model_name = 'SegNet'

sys_cfg.den_loss = 'mse_loss'
sys_cfg.att_loss = 'bce_loss'
sys_cfg.count_loss = 'global_loss'

# mc_drop threshold setting
sys_cfg.threshold_var = 0.0001
# sys_cfg.threshold_var
sys_cfg.p_weight = 1
sys_cfg.n_weight = 0.2
# uda loss setting
sys_cfg.unsup_wl = 0.5

sys_cfg.lr = 1e-4  # MCNN 1e-4  # CSRNet 1e-5
sys_cfg.num_epoch_lr_decay = 1
sys_cfg.lr_decay = 0.995
sys_cfg.max_epoch = 1000
sys_cfg.reduce_channel = 1

sys_cfg.seed = 1234
sys_cfg.cudnn_benchmark = False
sys_cfg.gpus = [0, 1]

# distiller
sys_cfg.lamb_fsp = 0
sys_cfg.lamb_cos = 0

# mean teacher
sys_cfg.mt_mode = 'mt'

# path
now = time.strftime("%m%d%H%M", time.localtime())
sys_cfg.exp_name = now + '_' + sys_cfg.dataset + '_' + sys_cfg.model_name
# sys_cfg.exp_log_dir = sys_cfg.exp_name
# print(sys_cfg.exp_name)
# sys_cfg.exp_dir = './exp'
sys_cfg.exp_dir = './exp_2022'
# sys_cfg.resume = False
# sys_cfg.work_dir = os.path.split(os.path.realpath(__file__))[0]
sys_cfg.work_dir = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

sys_cfg.log_freq = 10
sys_cfg.val_freq = 5
sys_cfg.val_dense_start = 500

# metric
sys_cfg.patch_level = 4
