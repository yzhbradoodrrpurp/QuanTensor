# -*- coding = utf-8 -*-
# @Time: 2025/4/13 15:33
# @Author: Zhihang Yi
# @File: Transformer.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR
import torch.optim as optim
import pandas as pd
from Data.Dataset import MyDataset


class Transformer(nn.Module):

    def __init__(self, d_model=4, nhead=4, num_layers=8, dtype=torch.float, device='cpu'):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dtype=dtype, device=device)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 4, dtype=dtype, device=device),
            nn.LayerNorm(d_model * 4, dtype=dtype, device=device),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model, dtype=dtype, device=device),
            nn.LayerNorm(d_model, dtype=dtype, device=device),
            nn.ReLU(),
            nn.Linear(d_model, 2, dtype=dtype, device=device),
        )

    def preprocess(self, path):
        csv = pd.read_csv(path)
        data = csv.loc[2:, ['Close', 'High', 'Low', 'Open']].astype(float)
        label = csv.loc[2:, 'Signal'].astype(float)

        X = torch.tensor(data.values, dtype=self.dtype, device=self.device)  # (N, 4)
        y = torch.tensor(label.values, dtype=torch.float, device=self.device)  # (N,)

        X_list = []
        y_list = []
        window = 36
        N = X.shape[0]

        for i in range(N - 36):
            X_sample = X[i:i + window, :]
            y_sample = y[i:i + window]

            X_list.append(X_sample)
            y_list.append(y_sample)

        X = torch.stack(X_list, dim=0)  # (N - window, window, 4)
        y = torch.stack(y_list, dim=0)  # (N - window, window)

        return X, y

    def forward(self, X):
        """

        :param X: (N, T, D)
        :return: scores: (N, 2)
        """
        mask = Transformer.get_mask(X.shape[1], dtype=self.dtype, device=self.device)
        feature_map = self.encoder(X, mask=mask)  # (N - window, window, D)
        scores = self.classifier(feature_map)  # (N - window, window, 2)

        return scores

    @staticmethod
    def get_mask(T, dtype, device):
        mask = torch.triu(torch.ones(T, T, dtype=dtype, device=device), diagonal=1)
        mask[mask == 1] = float('-inf')

        return mask


if __name__ == '__main__':
    model = Transformer(d_model=4, nhead=4, num_layers=8)

    path = '../Data/BTC/btc_train.csv'

    # X_train: (N - window, window, 4)
    # y_train: (N - window, window)
    X_train, y_train = model.preprocess(path)

    training_set = MyDataset(X_train, y_train)

    dataloader = DataLoader(training_set, batch_size=32, shuffle=False)

    model = Transformer(d_model=4, nhead=4, num_layers=12, dtype=torch.float, device='cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.96)
    epochs = range(100)

    scheduler = LinearLR(optimizer, start_factor=0.05, total_iters=10)

    model.train()

    for epoch in epochs:
        for X, y in dataloader:
            scores = model(X)  # (batch, window, 2)

            scores_flat = scores.view(-1, 2)  # (batch * window, 2)
            y_flat = y.view(-1)

            loss = criterion(scores_flat, y_flat.to(torch.long))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        torch.save(model.state_dict(), 'transformer.pth')

        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')
