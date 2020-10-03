"""
train_nonl.py
author:worith

data_driven

"""

from tensorboardX import SummaryWriter
import numpy as np
from utils.handler import get_data, split_data
from torch.utils.data import DataLoader, Dataset, TensorDataset
from model.net_f1 import Net
import torch
import time
from torch.autograd import Variable

BATCH_SIZE = 50


def main():
    data = get_data('data/oil_data.xlsx')
    # data=data[data['输入方向']>0]
    # data=data[data['支管与主管流量比Q1/Q0']==0.25]

    # raw features
    data = np.array(data, dtype=np.float32)[:, (6, 10)]
    # train_data=[]
    # test_data=[]
    # for i in range(data.shape[0]):
    #     if i%10==0:
    #         test_data.append(data[i])
    #     else:
    #         train_data.append(data[i])
    #
    # train_data=np.array(train_data,dtype=np.float32)
    # test_data=np.array(test_data,dtype=np.float32)

    # split train/test
    train_data, test_data = split_data(data, 0.1)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    net = Net(1, 4, 4, 1)

    # Adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # loss :MSE
    loss_func = torch.nn.MSELoss()

    # use tensorboardX
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    writer = SummaryWriter('./log/events' + timestamp + 'train_model_Q')
    iter = 0

    net.train()
    for epoch in range(500):
        for step, data in enumerate(train_loader):
            # x,y=Variable(x),Variable(y)
            x, y = data[:, 0].reshape(data.shape[0], 1), data[:, 1].reshape(data.shape[0], 1)
            # print(x)
            prediction = net(x)
            loss = loss_func(prediction, y)

            # Backward error propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter += 1

            # store loss
            writer.add_scalar('train_loss', loss.item(), iter)

    # stop backward

    net.eval()
    pre_loss = 0
    for step, data in enumerate(test_loader):
        x, y = data[:, 0].reshape(data.shape[0], 1), data[:, 1].reshape(data.shape[0], 1)
        prediction = net(x)
        pre_loss = loss_func(prediction, y)

        # store MSE of test
        writer.add_scalar('test_loss', pre_loss.item(), step)


if __name__ == '__main__':
    main()
