from torch.utils.data import DataLoader
from ccm.dataset.utils import TwoStreamBatchSampler
from ..DataTrans.Transforms import DeNormalize
import torch
from torch.utils.data.dataloader import default_collate


def custom_collate(batch):
    '''
    :param batch: [list] The length of batch is batch_size,
    batch: list, its length is 1. e.g., [dict{}]
    '''

    # transposed_batch = list(zip(*batch))
    data_dict = {}
    if isinstance(batch[0], dict):
        for key in batch[0]:
            if key == 'points' or key == 'name' or key == 'p_target':
                data_dict[key] = [d[key] for d in batch]
            elif key == 'is_label' or key == 'gd_count' or key == 'gt_count':
                data_dict[key] = torch.tensor([d[key] for d in batch])
            else:
                data_dict[key] = torch.stack([d[key] for d in batch], 0)
    return data_dict


def make_dataset(args):
    dataset_name = args.dataset.lower()
    if dataset_name == 'shha':
        from .SHHA.SHHA_Dataset import SHHADataset as Dataset
    elif dataset_name == 'shhb':
        from .SHHB.ShhbDataset import SHHBDataset as Dataset
    else:
        raise ValueError('unknown dataset, please check!!!')

    dataset_dict = {'train': Dataset(mode='train', args=args),
                    'val': Dataset(mode='val', args=args),
                    'test': Dataset(mode='test', args=args)}

    return dataset_dict


def make_dataloader(args, learning_mode='SSL'):
    """
    description: make dataloader for training, validation and testing
    :param args:
    :param learning_mode:
    :return:
    """

    dataset_dict = make_dataset(args)
    train_dataset = dataset_dict['train']
    val_dataset = dataset_dict['val']
    test_dataset = dataset_dict['test']

    labeled_idx, unlabeled_idx = train_dataset.get_labeled_unlabeled_idx()

    if 'eval' in args.dataset.lower():
        train_loader = DataLoader(train_dataset,
                                  collate_fn=custom_collate,
                                  batch_size=1,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    else:
        sampler = TwoStreamBatchSampler(primary_indices=unlabeled_idx,
                                        secondary_indices=labeled_idx,
                                        batch_size=args.batch_size,
                                        secondary_batch_size=args.label_batch_size)

        train_loader = DataLoader(train_dataset,
                                  collate_fn=custom_collate,
                                  batch_sampler=sampler,
                                  num_workers=args.num_workers,
                                  pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            collate_fn=custom_collate,
                            batch_size=1,
                            num_workers=args.num_workers,
                            pin_memory=True)

    test_loader = DataLoader(test_dataset,
                             collate_fn=custom_collate,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader
