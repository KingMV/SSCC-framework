import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='HPS')
    # SSL method
    parser.add_argument('--method', type=str, default='MT', help='the name of SSL or FL method')
    parser.add_argument('--config', type=str, default='./config/exp_cfg/MT.json', help='the config file of the method')

    # training parameters setting
    parser.add_argument('--lr', type=float, default=1e-4, help='the initial learning rate')
    # parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='the ratio of lr decay')
    # parser.add_argument('--num_epoch_lr_decay', type=int, default=1, help='the number of epoch to decay lr')
    # parser.add_argument('--max_epoch', type=int, default=500, help='the number of epoch to train')
    parser.add_argument('--batch_size', type=int, default=16, help='the total batch size')
    parser.add_argument('--label_batch_size', type=int, default=8, help='train batch size')

    # gpu setting
    parser.add_argument('--gpus', type=str, default='0', help='[0]')

    # dataset setting
    parser.add_argument('--dataset', type=str, default='SHHA',
                        help='the name of dataset: SHHA, SHHB, UCF50, UCF_CC_50, UCF_QNRF, NWPU ')

    # perturbation setting
    parser.add_argument('--perturb_way', type=str, default='None',
                        help='the name of perturbation: None')
    parser.add_argument('--perturb_region', type=str, default='None', help='xxxx')
    parser.add_argument('--perturb_method', type=str, default='None', help='xxxx')
    # parser.add_argument('--perturb_region', type=float, default=0.0, help='xxxx')

    # model
    parser.add_argument('--network', default='CSRNet', type=str, help='the name of model: CSRNet, MCNN')

    parser.add_argument('--note', type=str, default='Base', help='the note of this experiment')

    # # loss setting
    parser.add_argument('--con_w1', type=float, default=0.5, help='the weight of consistency loss for unlabeled data')

    args = parser.parse_args()

    return args
