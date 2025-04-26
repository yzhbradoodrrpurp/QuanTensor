# -*- coding = utf-8 -*-
# @Time: 2025/4/13 17:02
# @Author: Zhihang Yi
# @File: run.py
# @Software: PyCharm

import torch
from index import accuracy, trending
import Data
from Model.Transformer import Transformer


def run_accuracy():
    path = '../Model/trained_weights/transformer_BTC_classification.pth'
    model = Transformer(d_model=Data.dimension, nhead=4, num_layers=12, window=168, task='classification',
                        dtype=torch.float, device='cuda')
    model.load_state_dict(torch.load(path))

    train_path = '../Data/BTC/train.csv'
    eval_path = '../Data/BTC/val.csv'

    training_accuracy = accuracy(model, train_path)
    validation_accuracy = accuracy(model, eval_path)

    print(f'Training Accuracy: {training_accuracy:.4f}, Validation Accuracy: {validation_accuracy:.4f}')


def run_trending():
    path = '../Model/trained_weights/transformer_BTC_prediction.pth'
    model = Transformer(d_model=Data.dimension, nhead=4, num_layers=12, window=168, task='prediction',
                        dtype=torch.float, device='cuda')
    model.load_state_dict(torch.load(path))

    train_path = '../Data/BTC/train.csv'
    eval_path = '../Data/BTC/val.csv'

    trending(model, train_path)
    trending(model, eval_path)