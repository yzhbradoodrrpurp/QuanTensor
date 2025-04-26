# -*- coding = utf-8 -*-
# @Time: 2025/4/13 16:29
# @Author: Zhihang Yi
# @File: Dataset.py
# @Software: PyCharm

from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]