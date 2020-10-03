"""
train_all.py
author:worith
Hybrid model 1


"""

from tensorboardX import SummaryWriter
import numpy as np
from utils.handler import get_data, onehotHandler, normalizationHanlder, split_data
from torch.utils.data import DataLoader, Dataset, TensorDataset
from model.net_f1 import Net
import torch
import time
import pandas as pd
from torch.autograd import Variable

BATCH_SIZE = 359


def main():
    data = get_data('data/oil_data.xlsx')
    # one-hot encode
    one_hot_feature = ['输入方向']
    hot = onehotHandler(data, one_hot_feature)
    # normalization min-max
    regfeatures = ['流体速度U0', '流体粘度μf', '支撑剂密度ρs', '支撑剂粒径d50', '支管与主管流量比Q1/Q0', 'k,-0.05次方', '主管内固体浓度C0']
    norm_data = normalizationHanlder(data, regfeatures)
    # concat
    data = data.drop(columns=one_hot_feature)
    data = data.drop(columns=regfeatures)
    data = pd.concat([hot, norm_data, data], axis=1)
    # #z-score normalization
    # datas=np.array(data[features].apply(lambda x:(x-x.mean())/(x.std())))
    data = np.array(data[[0, 1, 2, '流体速度U0', '流体粘度μf', '支撑剂密度ρs', '支撑剂粒径d50', \
                          '支管与主管流量比Q1/Q0', '主管内固体浓度C0', 'k,-0.05次方', 'PTE']], dtype=np.float32)
    # train_data=[]    # test_data=[]
    # for i in range(data.shape[0]):
    #     if i%10==0:
    #         test_data.append(data[i])
    #     else:
    #         train_data.append(data[i])
    # 
    # train_data=np.array(train_data,dtype=np.float32)
    # test_data=np.array(test_data,dtype=np.float32)

    # split  train/test
    train_data, test_data = split_data(data, 0.1)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    net = Net(10, 4, 4, 1)

    # Adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # Loss MSE
    loss_func = torch.nn.MSELoss()
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # writer = SummaryWriter('./log/events'+timestamp+'train_model_ALL')
    ALLloss = []

    # global_step
    iter = 0
    net.train()
    for epoch in range(500):
        for step, data in enumerate(train_loader):
            # x,y=Variable(x),Variable(y)
            x, y = data[:, 0:10], data[:, 10].reshape(data.shape[0], 1)
            # print(x)
            prediction = net(x)
            loss = loss_func(prediction, y)

            # Backward error propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter += 1
            # writer.add_scalar('train_loss', loss.item(), iter)

    # store net.parameters
    # torch.save(net.state_dict(), 'pkl/data_driven_params.pkl')
    print("loss:", loss)
    net.eval()
    pre_loss = 0
    test_loss = []
    for step, data in enumerate(test_loader):
        x, y = data[:, 0:10], data[:, 10].reshape(data.shape[0], 1)
        prediction = net(x)
        pre_loss = loss_func(prediction, y).item()
        test_loss.append(pre_loss)
        # writer.add_scalar('test_loss',pre_loss,step)
    print("MSE_mean:", np.mean(test_loss), "MSE_std:", np.std(test_loss))

    # loss gap
    print(("gap:%f") % (np.mean(test_loss) - loss))


if __name__ == '__main__':
    main()
    # data=np.array(pd.read_csv('data/output/test_loss_wok.csv'))
    # mean=0.0
    # std=0.0
    # for i,iter in enumerate(data):
    #     mean+=np.mean(data)
    #     std+=np.var(data)
    # print('Mean:',mean/5,';Std:',std/5)
