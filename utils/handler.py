"""
handler.py
author:worith

utils/some functions

"""
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = 'SimHei,Times New Roman'
plt.rcParams['font.size'] = 18

random_seed = 1234


# print corr png
def plot_corr(df):
    pd.plotting.scatter_matrix(df, alpha=0.8, figsize=(10, 10), diagonal='hist')
    # plt.title(wtName)
    # plt.savefig(plotPath+wtName+'_corr.png',dpi=300)
    # plt.show()


def get_data(filename):
    data = pd.read_excel(filename, sheet_name=0)

    data.replace("upward", 0, inplace=True)
    data.replace("downward", 1, inplace=True)
    data.replace("side", 10, inplace=True)

    # data.plot(subplots=True, layout=(-1, 3), figsize=(18, 10), sharex=True)
    # # plt.savefig("../figures/dataFrames.png", bbox_inches='tight')
    # data=np.array(data[['输入方向','流体速度U0','流体粘度μf','支撑剂密度ρs','支撑剂粒径d50','支管与主管流量比Q1/Q0','主管内固体浓度C0','k,-0.05次方','PTE']])
    # data=pd.DataFrame(data,columns=['输入方向','速度${U_0}$','粘度${μ_f}$','密度${ρ_s}$','粒径${d_{50}}$','流量比Q1/Q0','浓度C0','${k^{-0.05}}$','PTE'])
    # plot_corr(data)
    # plt.savefig("../figures/corr.png")
    # print(data.corr())
    # data.corr().to_csv("../data/output/corr.csv", index=False, sep=',')
    #
    # plt.show()
    return data


# features one-hot encoder
def onehotHandler(data, oneHotfeatures):
    onehotEncoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    hot = onehotEncoder.fit_transform(data[oneHotfeatures])
    hot = pd.DataFrame(hot)
    # data=data.drop(columns=oneHotfeatures)
    # return pd.concat([hot,data],axis=1)
    return hot


# features normalization
def normalizationHanlder(data, normFeatures):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data[normFeatures])
    data = pd.DataFrame(data, columns=normFeatures)
    return data


# split train/test dataset
def split_data(data, size):
    train, test = train_test_split(data, test_size=size, random_state=random_seed)
    return train, test


def compute_corr(data, powers):
    datas = data[['惯性指数K', 'PTE']]
    datas['惯性指数K'] = np.power(datas['惯性指数K'], powers)
    corrs = datas.corr()
    return corrs


# hybrid model 3 : to get inter/coef of regularization
def get_AB(y, phi):
    phi = np.mat(phi)
    phi_T = np.transpose(phi)
    term = (phi_T * phi).I
    AB = term * phi_T * y.numpy()
    B = AB[-1]
    A = AB[0:-1]
    return A, B


def main():
    # print(data_xls.sheet_names[2])
    # data = data_xls.parse(sheetname=data_xls.sheet_names[0], header=None)
    # data=get_data('../data/oil_data.xlsx')

    # powers=np.linspace(-5,5,401)
    # corrs=[]
    # for x in powers:
    #    corr=np.array(compute_corr(data,x))
    #    corrs.append(corr[0][1])
    # print(corrs)
    # plt.plot(powers,corrs,'b-',lw=2)
    # plt.show()
    # data=onehotHandler(data,['输入方向','主管内固体浓度C0'])
    # print(data)
    pass

    # data = pd.read_excel("../data/8#.xlsx",None)
    # sheets=data.keys()
    # for sheet in sheets:
    #     data=pd.read_excel('../data/8#.xlsx',sheet_name=sheet)
    #     data_columns = data.columns
    #     data = np.array(data)[1:, 1:]
    #     data = pd.DataFrame(data, columns=data_columns[1:])
    #     data.plot(subplots=True, layout=(-1, 3), figsize=(18, 5), sharex=True)
    #     plt.savefig(("../figures/week_4/8#/%s.png") %(sheet), bbox_inches='tight')
    #     plt.show()

    # data = pd.read_excel("../data/1#.xlsm", sheet_name='Log Data (24)')
    # data_columns = data.columns
    # data = np.array(data)[1:, 1:]
    # data = pd.DataFrame(data, columns=data_columns[1:])
    # plot_corr(data)
    # plt.savefig("../figures/week_4/corr.png")
    # plt.show()
    #


if __name__ == '__main__':
    main()
