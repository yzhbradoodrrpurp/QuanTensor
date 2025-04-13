# -*- coding = utf-8 -*-
# @Time: 2025/4/13 17:07
# @Author: Zhihang Yi
# @File: index.py
# @Software: PyCharm

import torch
from torch.utils.data import DataLoader

def accuracy(model, dataset):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    acuracies = []

    for X, y in dataloader:
        scores = model(X)
        y_pred = scores.argmax(dim=1)

        compare = (y == y_pred)
        right = compare[compare == True].shape[0]
        total = y.shape[0]

        acuracies.append(right / total)

    return torch.tensor(acuracies).mean()
