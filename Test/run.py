# -*- coding = utf-8 -*-
# @Time: 2025/4/13 17:02
# @Author: Zhihang Yi
# @File: run.py
# @Software: PyCharm

import torch
from index import accuracy
from Data.Dataset import MyDataset
from Model.Transformer import Transformer

path = '../Model/transformer4BTC.pth'
model = Transformer(d_model=4, nhead=4, num_layers=12, dtype=torch.float, device='cpu')
model.load_state_dict(torch.load(path))

train_path = '../Data/BTC/train.csv'
eval_path = '../Data/BTC/val.csv'

X_train, y_train = model.preprocess(train_path)
X_val, y_val = model.preprocess(eval_path)

training_set = MyDataset(X_train, y_train)
validation_set = MyDataset(X_val, y_val)

training_accuracy = accuracy(model, training_set)
validation_accuracy = accuracy(model, validation_set)

print(f'Training Accuracy: {training_accuracy:.4f}, Validation Accuracy: {validation_accuracy:.4f}')
