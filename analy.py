"""
analy.py
author:worith

Sensitivity Analysis


"""

import torch
from utils.handler import get_data, normalizationHanlder, onehotHandler
from model.net_f1 import Net
from model.net_concat import net_concat, net_phy
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from train_concat import data_handler, get_physical_infos, net2
import time
from tensorboardX import SummaryWriter

# rc config
plt.rcParams['font.size'] = 18
plt.rcParams['font.sans-serif'] = 'Times New Roman'


# hybrid model 3: physical infos
def get_physical_infos(analy_loader):
    net = net_phy(1, 4, 4, 1)

    # restore net
    net.load_state_dict(torch.load('pkl/get_phy_params.pkl'))
    physical_infos = []
    net.eval()
    for step, data in enumerate(analy_loader):
        x_train, y_train = data[:, 9].reshape(data.shape[0], 1), data[:, 10].reshape(data.shape[0], 1)
        train_prediction, train_physical_info = net(x_train)
        physical_infos.append(train_physical_info)
    return physical_infos


# restore net
def retore_para(x, physical_infos, step):
    # net=Net(9,4,4,1)
    #
    # # 将保存的参数复制到 net3
    # net.load_state_dict(torch.load('pkl/data_driven_params.pkl'))
    # prediction = net(x)
    # return prediction

    # hybrid_2
    net = net_concat(9, 4, 8, 1)
    net.load_state_dict(torch.load('pkl/concat_params.pkl'))
    net.add_physical_info(physical_infos[step])
    prediction = net(x)
    return prediction


def main():
    data = get_data('data/oil_data.xlsx')
    data_delta1 = data
    data_delta2 = data
    # one-hot encode
    one_hot_feature = ['输入方向']
    hot = onehotHandler(data, one_hot_feature)
    # normalization min-max
    regfeatures = ['流体速度U0', '流体粘度μf', '支撑剂密度ρs', '支撑剂粒径d50', '支管与主管流量比Q1/Q0', '主管内固体浓度C0', 'k,-0.05次方']
    norm_data = normalizationHanlder(data, regfeatures)
    # concat
    data = data.drop(columns=one_hot_feature)
    data = data.drop(columns=regfeatures)
    data = pd.concat([hot, norm_data, data], axis=1)
    # #z-score normalization
    # datas=np.array(data[features].apply(lambda x:(x-x.mean())/(x.std())))
    data = np.array(data[[0, 1, 2, '流体速度U0', '流体粘度μf', '支撑剂密度ρs', '支撑剂粒径d50', \
                          '支管与主管流量比Q1/Q0', '主管内固体浓度C0', 'k,-0.05次方', 'PTE']], dtype=np.float32)
    # timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # writer = SummaryWriter('./log/events' + timestamp + 'train_model_Q')
    analy_loader = DataLoader(dataset=data, shuffle=False)

    # some bias
    delta1 = 0.5
    delta2 = -0.5

    # one-hot encoder
    hot_delta1 = onehotHandler(data_delta1, one_hot_feature)
    # normalization min-max
    norm_data_delta1 = normalizationHanlder(data_delta1, regfeatures)
    # concat
    data_delta1 = data_delta1.drop(columns=one_hot_feature)
    data_delta1 = data_delta1.drop(columns=regfeatures)
    data_delta1 = pd.concat([hot_delta1, norm_data_delta1, data_delta1], axis=1)
    for i, x in enumerate(data_delta1['支管与主管流量比Q1/Q0']):
        data_delta1['支管与主管流量比Q1/Q0'][i] = data_delta1['支管与主管流量比Q1/Q0'][i] + delta1

    data_delta1 = np.array(data_delta1[[0, 1, 2, '流体速度U0', '流体粘度μf', '支撑剂密度ρs', '支撑剂粒径d50', \
                                        '支管与主管流量比Q1/Q0', '主管内固体浓度C0', 'k,-0.05次方', 'PTE']], dtype=np.float32)
    # timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # writer = SummaryWriter('./log/events' + timestamp + 'train_model_Q')
    analy_delta1_loader = DataLoader(dataset=data_delta1, shuffle=False)

    hot_delta2 = onehotHandler(data_delta2, one_hot_feature)
    # normalization min-max
    norm_data_delta2 = normalizationHanlder(data_delta2, regfeatures)
    # concat
    data_delta2 = data_delta2.drop(columns=one_hot_feature)
    data_delta2 = data_delta2.drop(columns=regfeatures)
    data_delta2 = pd.concat([hot_delta2, norm_data_delta2, data_delta2], axis=1)

    for i, x in enumerate(data_delta2['支管与主管流量比Q1/Q0']):
        data_delta2['支管与主管流量比Q1/Q0'][i] = data_delta2['支管与主管流量比Q1/Q0'][i] + delta2

    data_delta2 = np.array(data_delta2[[0, 1, 2, '流体速度U0', '流体粘度μf', '支撑剂密度ρs', '支撑剂粒径d50', \
                                        '支管与主管流量比Q1/Q0', '主管内固体浓度C0', 'k,-0.05次方', 'PTE']], dtype=np.float32)
    # timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # writer = SummaryWriter('./log/events' + timestamp + 'train_model_Q')
    analy_delta2_loader = DataLoader(dataset=data_delta2, shuffle=False)

    # predict all dataset with different bias
    test_origin = []
    test_prediction1 = []
    test_prediction2 = []
    loss_func = torch.nn.MSELoss()
    for step, data1 in enumerate(analy_loader):
        x, conc, y = data1[:, 0:9], data1[:, 9].reshape(data1.shape[0], 1), data1[:, 10].reshape(data1.shape[0], 1)
        physical_infos = get_physical_infos(analy_loader)
        prediction = retore_para(x, physical_infos, step)

        pre_loss = loss_func(prediction, y).item()
        test_origin.append(prediction)
        # writer.add_scalar('test_loss', pre_loss, step)

    for step, data1 in enumerate(analy_delta1_loader):
        x, conc, y = data1[:, 0:9], data1[:, 9].reshape(data1.shape[0], 1), data1[:, 10].reshape(data1.shape[0], 1)
        physical_infos = get_physical_infos(analy_delta1_loader)
        prediction = retore_para(x, physical_infos, step)
        pre_loss = loss_func(prediction, y).item()
        test_prediction1.append(prediction)
        # writer.add_scalar('test_loss', pre_loss, step)

    for step, data1 in enumerate(analy_delta2_loader):
        # x, y = data1[:, 0:9], data1[:, 10].reshape(data1.shape[0], 1)
        # prediction = retore_para(x, analy_delta2_loader, step)
        x, conc, y = data1[:, 0:9], data1[:, 9].reshape(data1.shape[0], 1), data1[:, 10].reshape(data1.shape[0], 1)
        physical_infos = get_physical_infos(analy_delta2_loader)
        prediction = retore_para(x, physical_infos, step)
        pre_loss = loss_func(prediction, y).item()
        test_prediction2.append(prediction)
        # writer.add_scalar('test_loss', pre_loss, step)

    # plot figure
    plt.plot(np.arange(0, len(test_origin))[0:100], test_origin[0:100], 'r-', lw=2, label='origin Q1/Q0')
    plt.plot(np.arange(0, len(test_prediction1))[0:100], test_prediction1[0:100], 'b-', lw=2, label='add bias=+0.5')
    plt.plot(np.arange(0, len(test_prediction2))[0:100], test_prediction2[0:100], 'y-', lw=2, label='add bias=-0.5')
    plt.legend(loc='upper right')

    plt.ylabel('Predicted PTE')
    plt.xlabel('Samples')
    plt.savefig('figures/week_4/Hybrid_2.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
