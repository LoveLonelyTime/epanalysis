import random
import numpy as np
import torch
from torch.utils.data import Dataset

from logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class EpdDataset(Dataset):
    """
    加载数据集
    """
    def __init__(self, val_num=0.2, is_val=False):

        comsume_data_a = np.load('data/consume_data.npy')
        type_data_a = np.load('data/type_data.npy')

        data_len = len(comsume_data_a)

        if is_val:
            self.comsume_data = comsume_data_a[-int(data_len * val_num):]
            self.type_data = type_data_a[-int(data_len * val_num):]
        else:
            self.comsume_data = comsume_data_a[:int(data_len * (1 - val_num))]
            self.type_data = type_data_a[:int(data_len * (1 - val_num))]

        log.info(
            "{!r}: {} {} samples".format(
                self,
                len(self.comsume_data),
                "validation" if is_val else "training",
            )
        )

    def __len__(self):
        return len(self.comsume_data)

    def __getitem__(self, index):
        seq = torch.from_numpy(self.type_data[index]).to(dtype=torch.float32)
        out = torch.from_numpy(self.comsume_data[index]).to(
            dtype=torch.float32)
        return seq, out
