import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

plt.rcParams['font.sans-serif'] = 'SimHei,Times New Roman'
#plt.rcParams['font.size']=18



#print corr png
def plot_corr(df):
    pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(8, 8), diagonal='hist')
    # plt.title(wtName)
    # plt.savefig(plotPath+wtName+'_corr.png',dpi=300)
    #plt.show()



def get_data(filename):
    data=pd.read_excel(filename,sheet_name=0)

    data.replace("upward",00,inplace=True)
    data.replace("downward", 1, inplace=True)
    data.replace("side", 10, inplace=True)

    # data.plot(subplots=True, layout=(-1, 3), figsize=(18, 10), sharex=True)
    # plt.savefig("../figures/dataFrames.png", bbox_inches='tight')
    # plot_corr(data[['输入方向','支管与主管流量比Q1/Q0','主管内固体浓度C0','惯性指数K','k,-0.05次方','PTE']])
    # #plt.savefig("../figures/corr.png")
    # print(data.corr())
    # data.corr().to_csv("../data/output/corr.csv", index=False, sep=',')
    # plt.show()
    return data


# features one-hot encoder
def onehotHandler(data,oneHotfeatures):
    onehotEncoder=OneHotEncoder(sparse=False,handle_unknown="ignore")
    hot=onehotEncoder.fit_transform(data[oneHotfeatures])
    hot=pd.DataFrame(hot)
    data=data.drop(columns=oneHotfeatures)
    return pd.concat([hot,data],axis=1)

def main():
    data=get_data('../data/oil_data.xlsx')
    data=onehotHandler(data,['输入方向','主管内固体浓度C0'])

    print(data)

if __name__=='__main__':
    main()
