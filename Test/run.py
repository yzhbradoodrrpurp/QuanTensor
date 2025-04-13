# -*- coding = utf-8 -*-
# @Time: 2025/4/13 17:02
# @Author: Zhihang Yi
# @File: run.py
# @Software: PyCharm

import torch
from index import accuracy
from Data.Dataset import MyDataset
from Model.Transformer import Transformer

path = '../Model/transformer.pth'
model = torch.load(path, weights_only=False)

train_path = '../../Data/BTC/btc_train.csv'
eval_path = '../../Data/BTC/btc_val.csv'

X_train, y_train = model.preprocess(train_path)
X_val, y_val = model.preprocess(eval_path)

training_set = MyDataset(X_train, y_train)
validation_set = MyDataset(X_val, y_val)

training_accuracy = accuracy(model, training_set)
validation_accuracy = accuracy(model, validation_set)

print(training_accuracy)
print(validation_accuracy)

