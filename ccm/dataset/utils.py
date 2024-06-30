import itertools
import numpy as np
from torch.utils.data.sampler import Sampler


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        # super(self, TwoStreamBatchSampler).__init__(None)
        self.primary_indices = primary_indices  # 无标签数据
        self.secondary_indices = secondary_indices  # 有标签数据
        self.secondary_batch_size = secondary_batch_size  # 有标签数据的batch_size
        self.primary_batch_size = batch_size - secondary_batch_size  # 无标签数据的batch_size
        if self.primary_batch_size != 0 and self.secondary_batch_size != 0:  #
            assert len(self.primary_indices) >= self.primary_batch_size > 0  # 无标签数据总数大于无标签数据的batch_size
            assert len(self.secondary_indices) >= self.secondary_batch_size > 0  # 有标签数据总数大于有标签数据的batch_size

    def __iter__(self):
        if self.primary_batch_size == 0:  # 无标签数据的batch_size为0
            # unlabel data is 0
            secondary_iter = iterate_once(self.secondary_indices)  # 只使用有标签数据的迭代器
            return (
                secondary_batch for secondary_batch in grouper(secondary_iter, self.secondary_batch_size)
            )
        elif self.secondary_batch_size == 0:  # 有标签数据的batch_size为0
            primary_iter = iterate_once(self.primary_indices)
            return (
                primary_batch for primary_batch in grouper(primary_iter, self.primary_batch_size)
            )
        else:
            if len(self.primary_indices) > len(self.secondary_indices):  # 无标签数据总数大于有标签数据总数
                primary_iter = iterate_once(self.primary_indices)  # 使用无标签数据作为主迭代器（只迭代一次）
                # print(primary_iter)
                secondary_iter = iterate_eternally(self.secondary_indices)  # 使用有标签数据作为辅迭代器（无限迭代）
            else:
                # print(1111111111)
                secondary_iter = iterate_once(self.secondary_indices)
                primary_iter = iterate_eternally(self.primary_indices)
            return (
                primary_batch + secondary_batch
                for (primary_batch, secondary_batch)
                in zip(grouper(primary_iter, self.primary_batch_size),
                       grouper(secondary_iter, self.secondary_batch_size))
            )

    def __len__(self):
        if self.primary_batch_size == 0:
            return len(self.secondary_indices) // self.secondary_batch_size
        elif self.secondary_batch_size == 0:
            return len(self.primary_indices) // self.primary_batch_size
        elif len(self.primary_indices) >= len(self.secondary_indices):
            return len(self.primary_indices) // self.primary_batch_size
        elif len(self.primary_indices) < len(self.secondary_indices):
            return len(self.secondary_indices) // self.secondary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)  # 打乱序列


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def relabel_dataset(dataset):
    unlabeled_idxs = []
    for idx in range(len(dataset.data)):
        if dataset.data[idx][1] == -1:  # img, label, edge = dataset.imgs[idx]
            unlabeled_idxs.append(idx)
    labeled_idxs = sorted(set(range(len(dataset.data))) - set(unlabeled_idxs))

    return labeled_idxs, unlabeled_idxs
