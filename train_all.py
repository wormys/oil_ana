from tensorboardX import SummaryWriter
import numpy as np
from utils.handler import get_data,onehotHandler
from torch.utils.data import DataLoader,Dataset,TensorDataset
from model.net_f1 import Net
import torch
import time
from torch.autograd import Variable
BATCH_SIZE=50

def main():
    data=get_data('data/oil_data.xlsx')
    one_hot_feature=['输入方向','主管内固体浓度C0']
    data=onehotHandler(data,one_hot_feature)
    features=['支管与主管流量比Q1/Q0','k,-0.05次方','PTE']

    # #z-score normalization
    # datas=np.array(data[features].apply(lambda x:(x-x.mean())/(x.std())))
    data=np.array(data[features])

    train_data=[]
    test_data=[]
    for i in range(data.shape[0]):
        if i%10==0:
            test_data.append(data[i])
        else:
            train_data.append(data[i])

    train_data=np.array(train_data,dtype=np.float32)
    test_data=np.array(test_data,dtype=np.float32)
    train_loader=DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=False)
    test_loader=DataLoader(dataset=test_data,batch_size=1,shuffle=False)

    net=Net(2,4,4,1)
    optimizer=torch.optim.Adam(net.parameters(),lr=0.001)
    loss_func=torch.nn.MSELoss()
    timestamp = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    writer = SummaryWriter('./log/events'+timestamp+'train_model_ALLWoD')
    iter=0
    net.train()
    for epoch in range(500):
         for step,data in enumerate(train_loader):
             # x,y=Variable(x),Variable(y)
             x, y = data[:, 0:2], data[:, 2].reshape(data.shape[0], 1)
             #print(x)
             prediction=net(x)
             loss=loss_func(prediction,y)
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()
             iter+=1
             writer.add_scalar('train_loss', loss.item(), iter)

    net.eval()
    pre_loss=0
    for step,data in enumerate(test_loader):
         x, y = data[:, 0:2], data[:, 2].reshape(data.shape[0], 1)
         prediction=net(x)
         pre_loss=loss_func(prediction,y)
         writer.add_scalar('test_loss',pre_loss.item(),step)


if __name__ =='__main__':
    main()

