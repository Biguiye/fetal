import os
import sys
from datetime import datetime
import torch
import numpy as np
from sklearn import metrics
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import torch.utils.data as Data

batch_size = 4
start_epoch = 1

N_epochs = 100
model_lr = 1e-2
lr_milestones = [20]
lr_gamma = 1

L2_factor = 4e-3

num_workers = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 交叉熵损失
criterion = torch.nn.CrossEntropyLoss()


class Net(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1 = nn.Sequential(
            # nn.Linear(in_features, 2, bias=False),
            # nn.Linear(2, out_features, bias=False),
            nn.Linear(in_features, out_features, bias=False),
        )

    def forward(self, x):
        return self.layer1(x)


def train(model, device, train_loader, optimizer):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    print('all train sample: ', total_num, ' / batch num: ', len(train_loader))

    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device, non_blocking=True), Variable(target).to(device, non_blocking=True)

        output = model(data)
        # 交叉熵损失
        loss = criterion(output, target.to(torch.int64))

        _, pred = torch.max(output.data, 1)
        correct += torch.sum(pred == target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss on train : {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
    acc = (correct / len(train_loader.dataset)).item()
    ave_loss = sum_loss / len(train_loader)

    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        ave_loss, correct, len(train_loader.dataset), 100 * acc))


if __name__ == '__main__':

    # torch.cuda.set_device(0)

    model = Net(in_features=2, out_features=2)
    model.to(device)

    # Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=model_lr, weight_decay=L2_factor)
    # 学习率多步退火
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)

    # train_features = np.array([
    #     [-0.1, 1],
    #     [0.2, 1.2],
    #     [1.1, -0.2],
    #     [1.2, 0.1],
    # ])
    train_features = np.array([
        [5, 100],
        [3, 200],
        [100, 2],
        [200, 199],
    ])

    train_label = np.array([0, 0, 1, 1])

    train_dataset = Data.TensorDataset(
        torch.from_numpy(train_features).float(), torch.from_numpy(train_label).long())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, N_epochs + 1):
        print('epoch : ', epoch)
        train(model, device, train_loader, optimizer)

        # 阶段式退火
        scheduler.step()
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']

    print(model)

    model_dict = model.state_dict()

    for k in model_dict.keys():
        print(k, ' : ', model_dict[k])
