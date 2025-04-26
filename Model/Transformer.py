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
import Data
from Data.Dataset import MyDataset
import logging
from datetime import datetime
import os


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, window, dtype=torch.float, device='cpu'):
        super().__init__()

        self.d_model = d_model
        self.window = window
        self.dtype = dtype
        self.device = device

        self.embedding = nn.Embedding(window, d_model, dtype=dtype, device=device)

    def forward(self, X):
        """

        Args:
            X: (N, T, D)

        Returns:

        """
        idx = torch.arange(self.window, device=self.device)  # (T,)
        position = self.embedding(idx)  # (T, D)
        return X + position


class Transformer(nn.Module):

    def __init__(self, d_model=4, nhead=4, num_layers=8, window=36, task='classification', dtype=torch.float, device='cpu'):

        logging.info('Start to initialize Transformer.')

        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device
        self.window = window
        self.task = task

        if self.task == 'classification':
            out_dimension = 3
        elif self.task == 'prediction':
            out_dimension = 1
        else:
            raise ValueError(f'Task {self.task} not supported.')

        self.positional_encoding = PositionalEncoding(d_model, window, dtype, device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dtype=dtype, device=device)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 4, dtype=dtype, device=device),
            nn.LayerNorm(d_model * 4, dtype=dtype, device=device),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model, dtype=dtype, device=device),
            nn.LayerNorm(d_model, dtype=dtype, device=device),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(d_model, out_dimension, dtype=dtype, device=device),
        )

        logging.info('Finish initializing Transformer.')

    def preprocess(self, path):

        logging.info(f'Start preprocessing {path}.')

        csv = pd.read_csv(path)
        data = csv.iloc[1:, :Data.dimension].astype(float)

        if self.task == 'classification':
            label = csv.iloc[1:, :].loc[:, 'buy_sell'].astype(float)
        elif self.task == 'prediction':
            label = csv.iloc[1:, :].loc[:, 'next_close_price']
        else:
            raise ValueError('Invalid task.')

        X = torch.tensor(data.values, dtype=self.dtype, device=self.device)  # (N, 15)
        y = torch.tensor(label.values, dtype=torch.float, device=self.device)  # (N,)

        X_list = []
        y_list = []
        window = self.window
        N = X.shape[0]

        for i in range(N - window):
            X_sample = X[i:i + window, :]
            y_sample = y[i:i + window]

            X_list.append(X_sample)
            y_list.append(y_sample)

        X = torch.stack(X_list, dim=0)  # (N - window, window, 15)
        y = torch.stack(y_list, dim=0)  # (N - window, window)

        logging.info(f'Finish preprocessing {path}.')

        return X, y

    def forward(self, X):
        """

        :param X: (N, T, D)
        :return: scores: (N, 2)
        """
        logging.info('Start positional encoding.')
        feature_map = self.positional_encoding(X)
        logging.info('Finish positional encoding.')

        logging.info('Start masking the input data.')
        mask = Transformer.get_mask(feature_map.shape[1], dtype=self.dtype, device=self.device)
        logging.info('Finish masking the input data.')

        logging.info('Start encoding the input data.')
        feature_map = self.encoder(feature_map, mask=mask)  # (N - window, window, D)
        logging.info('Finish encoding the input data.')

        logging.info('Start classifying the input data.')
        scores = self.classifier(feature_map)  # (N - window, window, out_dimension)
        logging.info('Finish computing the scores.')

        return scores

    @staticmethod
    def get_mask(T, dtype, device):
        mask = torch.triu(torch.ones(T, T, dtype=dtype, device=device), diagonal=1)
        mask[mask == 1] = float('-inf')

        return mask


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s\t%(levelname)s\t%(message)s',
    datefmt='%m-%d-%Y %I:%M:%S',
    handlers=[
        logging.FileHandler('log/transformer.log'),
        # logging.StreamHandler()
    ]
)

if __name__ == '__main__':

    dtype = torch.float
    device = 'cuda'
    kind = 'BTC'

    # task = input('Specify the task:\nclassification\tprediction\n')
    task = 'prediction'

    model = Transformer(d_model=Data.dimension, nhead=4, num_layers=12, window=168, task=task, dtype=dtype, device=device)

    trained_weights_path = f'trained_weights/transformer_{kind}_{model.task}.pth'

    if os.path.exists(trained_weights_path):
        model.load_state_dict(torch.load(trained_weights_path))

    training_path = f'../Data/{kind}/train.csv'

    # X_train: (N - window, window, d)
    # y_train: (N - window, window)
    X_train, y_train = model.preprocess(training_path)
    training_set = MyDataset(X_train, y_train)
    dataloader = DataLoader(training_set, batch_size=128, shuffle=False)

    if task == 'classification':
        criterion = nn.CrossEntropyLoss()
        lr = 1e-6
    elif task == 'prediction':
        criterion = nn.MSELoss()
        lr = 20
    else:
        raise ValueError('Invalid task.')

    optimizer = optim.Adam(model.parameters(), lr=lr/100)
    # scheduler = LinearLR(optimizer, start_factor=0.05, total_iters=10)
    epochs = range(200)

    model.train()

    for epoch in epochs:
        for X, y in dataloader:
            scores = model(X)  # (batch, window, out_dimension)

            scores_flat = scores.view(scores.shape[0] * scores.shape[1], -1)  # (batch * window, out_dimension)
            y_flat = y.view(-1)

            if task == 'classification':
                loss = criterion(scores_flat, y_flat.to(torch.long))
            elif task == 'prediction':
                scores_flat = scores_flat.view(-1)
                loss = criterion(scores_flat, y_flat)
            else:
                raise ValueError('Invalid task.')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # scheduler.step()

        torch.save(model.state_dict(), trained_weights_path)

        logging.info(f'Epoch: {epoch}, Loss: {loss.item():.4f}')
        print(f'Time: {datetime.now().strftime("%y-%m-%d, %H:%M:%S")}, Epoch: {epoch}, Loss: {loss.item():.4f}')
