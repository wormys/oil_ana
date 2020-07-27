from tensorboardX import SummaryWriter
import numpy as np
from utils.handler import get_data,onehotHandler,normalizationHanlder,split_data
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
    data = get_data('data/oil_data.xlsx')
    # one-hot encode
    one_hot_feature = ['输入方向']
    hot = onehotHandler(data, one_hot_feature)
    # normalization min-max
    regfeatures = ['流体速度U0', '流体粘度μf', '支撑剂密度ρs', '支撑剂粒径d50', '支管与主管流量比Q1/Q0', '主管内固体浓度C0']
    norm_data = normalizationHanlder(data, regfeatures)
    # concat
    data = data.drop(columns=one_hot_feature)
    data = data.drop(columns=regfeatures)
    data = pd.concat([hot, norm_data, data], axis=1)
    # #z-score normalization
    # datas=np.array(data[features].apply(lambda x:(x-x.mean())/(x.std())))
    data = np.array(data[[0, 1, 2, '流体速度U0', '流体粘度μf', '支撑剂密度ρs', '支撑剂粒径d50', \
                          '支管与主管流量比Q1/Q0', '主管内固体浓度C0','k,-0.05次方','PTE']], dtype=np.float32)
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

    train_data, test_data = split_data(data, 0.1)
    features=['方向1','方向2','方向3','速度U0','粘度','密度','粒径','流量比','固体浓度C0','${k^{-0.05}}$','PTE']
    # features=['方向1','方向2','方向3','浓度1','浓度2','浓度3','流量比','${k^{-0.05}}$','PTE']
    # train_data=[]
    # test_data=[]
    # for i in range(datas.shape[0]):
    #     if i%10==0:
    #         test_data.append(datas[i])
    #     else:
    #         train_data.append(datas[i])
    model = xgb.XGBRegressor(learning_rate=0.001, n_estimators=500)
    train_data=pd.DataFrame(np.array(train_data,dtype=np.float32),columns=features)
    test_data=np.array(test_data,dtype=np.float32)
    tests=features[0:10]
    #dataframe  plot features
    model.fit(train_data[features[0:10]],train_data[features[10]])
    #pre=model.predict(test_data[:,0:4])
    plot_importance(model,importance_type='weight',max_num_features=10)
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    #yticks vertical center
    #plt.yticks(rotation='vertical',verticalalignment='center',fontsize=10)
    plt.savefig('figures/feature_importance.png',bbox_inches='tight')
    plt.show()

if __name__ =='__main__':
    main()

