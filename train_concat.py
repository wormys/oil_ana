from tensorboardX import SummaryWriter
import numpy as np
from utils.handler import get_data,onehotHandler,normalizationHanlder,split_data
from torch.utils.data import DataLoader,Dataset,TensorDataset
from model.net_f1 import Net
from model.net_concat import net_concat,net_phy
import torch
import time
import pandas as pd
from torch.autograd import Variable
BATCH_SIZE=50

def main():
    data=get_data('data/oil_data.xlsx')
    # one-hot encode
    one_hot_feature=['输入方向']
    hot=onehotHandler(data,one_hot_feature)
    # normalization min-max
    regfeatures=['流体速度U0','流体粘度μf','支撑剂密度ρs','支撑剂粒径d50','支管与主管流量比Q1/Q0','主管内固体浓度C0','k,-0.05次方']
    norm_data=normalizationHanlder(data,regfeatures)
    # concat
    data=data.drop(columns=one_hot_feature)
    data=data.drop(columns=regfeatures)
    data=pd.concat([hot,norm_data,data],axis=1)
    # #z-score normalization
    # datas=np.array(data[features].apply(lambda x:(x-x.mean())/(x.std())))
    data=np.array(data[[0,1,2,'流体速度U0','流体粘度μf','支撑剂密度ρs','支撑剂粒径d50',\
                        '支管与主管流量比Q1/Q0','主管内固体浓度C0','k,-0.05次方','PTE']],dtype=np.float32)
    # train_data=[]    # test_data=[]
    # for i in range(data.shape[0]):
    #     if i%10==0:
    #         test_data.append(data[i])
    #     else:
    #         train_data.append(data[i])
    #
    # train_data=np.array(train_data,dtype=np.float32)
    # test_data=np.array(test_data,dtype=np.float32)

    train_data, test_data = split_data(data, 0.1)
    train_loader=DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=False)
    test_loader=DataLoader(dataset=test_data,batch_size=1,shuffle=False)


    net1=net_phy(1,4,4,1)
    optimizer_net1=torch.optim.Adam(net1.parameters(),lr=0.001)
    loss_func=torch.nn.MSELoss()
    #timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    #writer = SummaryWriter('./log/events'+timestamp+'train_model_ALLWithoutK')
    ALLloss=[]
    train_physical_infos=[]
    test_physical_infos = []
    iter=0
    net1.train()
    for epoch in range(500):
         for step,data in enumerate(train_loader):
             # x,y=Variable(x),Variable(y)
             x, y = data[:, 9].reshape(data.shape[0],1), data[:, 10].reshape(data.shape[0], 1)
             #print(x)
             prediction=net1(x)
             loss=loss_func(prediction,y)
             optimizer_net1.zero_grad()
             loss.backward()
             optimizer_net1.step()
             iter+=1
             #writer.add_scalar('train_loss', loss.item(), iter)

    net1.eval()
    pre_loss=0
    test_loss=[]
    for step,data in enumerate(train_loader):
         x_train, y_train = data[:, 9].reshape(data.shape[0],1), data[:, 10].reshape(data.shape[0], 1)
         train_prediction,train_physical_info=net1(x_train)
         train_physical_infos.append(train_physical_info)
         # pre_loss=loss_func(train_prediction,y_train).item()
         # test_loss.append(pre_loss)

         #writer.add_scalar('test_loss',pre_loss.item(),step)
    for step,data in enumerate(test_loader):
         x_test, y_test = data[:, 9].reshape(data.shape[0],1), data[:, 10].reshape(data.shape[0], 1)
         test_prediction, test_physical_info = net1(x_test)
         test_physical_infos.append(test_physical_info)
         #writer.add_scalar('test_loss',pre_loss.item(),step)

    net2 = net_concat(9, 4, 8, 1)
    optimizer_net2 = torch.optim.Adam(net2.parameters(), lr=0.001)
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    writer = SummaryWriter('./log/events'+timestamp+'train_model_concat')
    iter = 0
    net2.train()
    for epoch in range(500):
        for step, data in enumerate(train_loader):
            # x,y=Variable(x),Variable(y)
            x, y = data[:, 0:9], data[:, 10].reshape(data.shape[0], 1)
            # print(x)
            net2.add_physical_info(train_physical_infos[step])
            prediction = net2(x)
            loss = loss_func(prediction, y)
            optimizer_net2.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_net2.step()
            iter += 1
            writer.add_scalar('train_loss', loss.item(), iter)

    net2.eval()
    pre_loss = 0
    test_loss = []
    for step, data in enumerate(test_loader):
        x, y = data[:, 0:9], data[:, 10].reshape(data.shape[0], 1)
        net2.add_physical_info(test_physical_infos[step])
        prediction = net2(x)
        pre_loss = loss_func(prediction, y)
        test_loss.append(pre_loss)
        writer.add_scalar('test_loss',pre_loss.item(),step)



if __name__ =='__main__':
     main()
