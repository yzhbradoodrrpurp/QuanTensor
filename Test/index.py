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
        scores_flat = scores.view(-1, 2)
        y_pred_flat = scores_flat.argmax(dim=1)  # (batch,)
        y_flat = y.view(-1)  # (batch,)

        compare = (y_flat == y_pred_flat)
        right = compare[compare == True].shape[0]
        total = y_flat.shape[0]

        acuracies.append(right / total)

    return torch.tensor(acuracies).mean().item()
