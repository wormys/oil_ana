from tensorboardX import SummaryWriter
import numpy as np
from utils.handler import get_data
from torch.utils.data import DataLoader,Dataset,TensorDataset
from model.net_f1 import Net
import torch
import time
import xgboost as xgb
import pandas as pd
from xgboost import plot_importance
import matplotlib.pyplot as plt
from torch.autograd import Variable
BATCH_SIZE=50

plt.rcParams['font.sans-serif']='SimHei,Times New Roman'
plt.rcParams['font.size']=15

def main():
    data=get_data('data/oil_data.xlsx')
    features=['输入方向','支管与主管流量比Q1/Q0','主管内固体浓度C0','k,-0.05次方','PTE']
    model=xgb.XGBRegressor(learning_rate=0.001,n_estimators=500,max_depth=5)
    #z-score normalization
    datas=np.array(data[features].apply(lambda x:(x-x.mean())/(x.std())))

    features=['方向','流量比','${k^{-0.05}}$','浓度C0','PTE']
    train_data=[]
    test_data=[]
    for i in range(datas.shape[0]):
        if i%10==0:
            test_data.append(datas[i])
        else:
            train_data.append(datas[i])

    train_data=pd.DataFrame(np.array(train_data,dtype=np.float32),columns=features)
    test_data=np.array(test_data,dtype=np.float32)
    #dataframe  plot features
    model.fit(train_data[features[0:4]],train_data[features[4]])
    #pre=model.predict(test_data[:,0:4])
    plot_importance(model)
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    #yticks vertical center
    plt.yticks(rotation='vertical',verticalalignment='center')
    plt.savefig('figures/feature_importance.png',bbox_inches='tight')
    plt.show()

if __name__ =='__main__':
    main()
