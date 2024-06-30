from torch.utils import data
import glob
import os
import h5py
import numpy as np
from PIL import Image
from ccm.DataTrans.Transforms import BaseTransform, TestTransform
import random
import scipy.io as sio
from .SHHA_Config import SHHA_CFG as cfg
from easydict import EasyDict as edict


class SHHADataset(data.Dataset):
    def __init__(self, mode, args=None):

        self.dd_args = cfg.copy()
        self.dd_args.update(args)
        self.dd_args = edict(self.dd_args)
        # print(self.dd_args)
        # print(self.dd_args.ROOT)

        self.mode = mode
        if self.mode not in ['train', 'val', 'test']:
            raise Exception("not implement")
        dataset = os.path.join('ShanghaiTech', 'part_A')

        self.data = []
        self.img_path_list = []
        self.val_img_list = []

        if self.mode == 'train' or self.mode == 'val':
            train_test_root = 'train_data'
        elif self.mode == 'test':
            train_test_root = 'test_data'
        else:
            train_test_root = None

        self.img_path_list = glob.glob(os.path.join(cfg.ROOT, dataset, train_test_root, 'images', '*.jpg'))
        random.shuffle(self.img_path_list)

        total_data_num = len(self.img_path_list)

        # if self.mode == 'train' or self.mode == 'val':
        #     num_val_data = int(total_data_num * args.data_split[2])

        if self.mode == 'train':
            num_label_data = int(total_data_num * args.data_split[0])
            num_unlabel_data = int(total_data_num * args.data_split[1])
            self.label_img_list = self.img_path_list[0:num_label_data]
            self.unlabel_img_list = self.img_path_list[num_label_data:num_label_data + num_unlabel_data]
            for img_name in self.label_img_list:
                self.data.append((img_name, 1))
            for img_name in self.unlabel_img_list:
                self.data.append((img_name, -1))
            # assert len(self.data) == len(self.img_path_list)
            print('[SHHA training set] labeled data has {0} and unlabeled data has {1}'.format(len(self.label_img_list),
                                                                                               len(self.unlabel_img_list)))

        elif self.mode == 'val':
            num_val_data = int(total_data_num * args.data_split[2])
            self.val_img_list = random.sample(self.img_path_list[num_val_data:], num_val_data)
            for img_name in self.val_img_list:
                self.data.append((img_name, 1))
            print('[SHHA val set] labeled data has {0}'.format(len(self.val_img_list)))

        elif self.mode == 'test':
            for img_name in self.img_path_list:
                self.data.append((img_name, 1))
            print('[SHHA test set] Total number of testing images: {0}'.format(len(self.img_path_list)))

        # transformation
        if self.mode == 'train':
            if self.dd_args.method == "STT":
                # self.trans = Semantic_Pertubration(self.dd_args)
                pass
            else:
                self.trans = BaseTransform(self.dd_args)
        else:
            self.trans = TestTransform(self.dd_args)

        self.load_memory = False
        self.load_memory_finish = False

    def __getitem__(self, index):
        samples = read_data(self.data[index])
        return self.trans(samples)

    def __len__(self):
        return len(self.data)

    def get_labeled_unlabeled_idx(self):
        unlabeled_idx = []
        labeled_idx = []
        if self.mode == 'train':
            for idx in range(len(self.data)):
                if self.data[idx][1] == -1:
                    unlabeled_idx.append(idx)
            labeled_idx = sorted(set(range(len(self.data))) - set(unlabeled_idx))
        # print('labeled_idx:{0}, unlabeled_idx:{1}'.format(len(labeled_idx), len(unlabeled_idx)))
        return labeled_idx, unlabeled_idx


def read_data(data):
    '''
    description: read data from the given path
    :param data: data-->[img_path,is_label] eg.['XXX.img',1]
    :return:
    '''
    data_dict = dict()
    filename, is_label = data
    image_name = os.path.basename(os.path.splitext(filename)[0])
    data_dict['name'] = image_name
    image = Image.open(filename).convert('RGB')

    # load keypoint
    gd_path = os.path.dirname(filename).replace('images', 'ground-truth')
    gd_file_name = 'GT_{}.mat'.format(image_name)
    gd_path = os.path.join(gd_path, gd_file_name)
    key_points = sio.loadmat(gd_path)['image_info'][0][0][0][0][0]

    # load density map
    den_name = 'GT_' + image_name + '.h5'
    den_path = os.path.join(os.path.dirname(filename).split('images')[0], 'sigma_5_g', 'gt_den', den_name)
    h5file = h5py.File(den_path, 'r')
    den_map = np.array(h5file['den'], dtype=np.float32)
    att_mask = (den_map > 0) * 1.0

    data_dict['den'] = den_map
    data_dict['att_mask'] = att_mask
    data_dict['is_label'] = is_label
    data_dict['image'] = image
    data_dict['points'] = key_points

    return data_dict
