from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from utils.handler import get_data,split_data
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']='SimHei, Times New Roman'
plt.rcParams['font.size']=18
def main():
    data=get_data('data/oil_data.xlsx')
    model = LinearRegression()
    data = np.array(data, dtype=np.float32)[:, (6, 10)]

    # train_data = []
    # test_data = []
    # for i in range(data.shape[0]):
    #     if i % 10 == 0:
    #         test_data.append(data[i])
    #     else:
    #         train_data.append(data[i])
    #
    # train_data = np.array(train_data, dtype=np.float32)
    # test_data = np.array(test_data, dtype=np.float32)

    train_data,test_data=split_data(data,0.1)
    x_train_data=np.array(train_data[:,0]).reshape((-1,1))
    x_test_data=np.array(test_data[:,0]).reshape((-1,1))
    y_train_data=np.array(train_data[:,1])
    y_test_data=np.array(test_data[:,1])

    model=model.fit(x_train_data,y_train_data)
    r_sq = model.score(x_train_data,y_train_data )

    print('coefficient of determination(𝑅²) :', r_sq)
    # coefficient of determination(𝑅²) : 0.715875613747954
    print('intercept:', model.intercept_)
    # （标量） 系数b0 intercept: 5.633333333333329 -------this will be an array when y is also 2-dimensional
    print('slope:', model.coef_)

    y_pred=model.predict(x_test_data)
    plt.plot(np.arange(y_pred.shape[0]),y_pred,'g--',lw=2,label='${y_{prec}}$')
    plt.plot(np.arange(y_pred.shape[0]),y_test_data,'r-',lw=2,label='${y_{true}}$')
    plt.legend()
    plt.show()

    print('predicted response:', y_pred, sep='\n')
    print('Exact response:', y_test_data, sep='\n')
    print('testMSE:',np.linalg.norm(y_test_data-y_pred,2))
    # print(x_data,y_data)



if __name__=='__main__':
    main()
