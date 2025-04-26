# -*- coding = utf-8 -*-
# @Time: 2025/4/13 17:07
# @Author: Zhihang Yi
# @File: index.py
# @Software: PyCharm

import torch
from torch.utils.data import DataLoader
import pandas as pd
from Data.Dataset import MyDataset
import matplotlib
import matplotlib.pyplot as plt


# matplotlib.use('TkAgg')


@torch.no_grad()
def accuracy(model, path):
    """

    :param model: transformer etc.
    :param path: training path or validation path
    :return:
    """
    model.eval()

    inputs, outputs = model.preprocess(path)
    dataset = MyDataset(inputs, outputs)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    acuracies = []

    for X, y in dataloader:
        scores = model(X)
        scores_flat = scores.view(-1, 3)
        y_pred_flat = scores_flat.argmax(dim=1)  # (batch,)
        y_flat = y.view(-1)  # (batch,)

        compare = (y_flat == y_pred_flat)
        right = compare[compare == True].shape[0]
        total = y_flat.shape[0]

        acuracies.append(right / total)

    return torch.tensor(acuracies).mean().item()


@torch.no_grad()
def trending(model, path):
    """

    :param model: transformer etc.
    :param path: training path or validation path
    :return:
    """
    model.eval()

    inputs, outputs = model.preprocess(path)
    dataset = MyDataset(inputs, outputs)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    real = pd.read_csv(path).loc[:, 'close']
    real = real.iloc[:-169]
    predicted = torch.tensor([]).to('cuda')

    x = list(range(real.shape[0]))

    for X, y in dataloader:
        scores = model(X)  # (N, T, 1)
        scores_flat = scores[:, -1, 0]
        predicted = torch.cat([predicted, scores_flat], dim=0)

    predicted = predicted.cpu().numpy()

    plt.figure()
    plt.plot(x, real, label='real', color='blue')
    plt.plot(x, predicted, label='predicted', color='orange')
    plt.legend()
    plt.savefig('plot.jpg')





