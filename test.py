import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression   # 线性回归
from sklearn.neighbors import KNeighborsRegressor   # K近邻回归
from sklearn.tree import DecisionTreeRegressor      # 决策树回归
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归
import lightgbm as lgb                              # LightGBM模型

from sklearn.metrics import mean_squared_error


# 读取训练数据和测试数据
def read_data():
    train_data_file = 'zhengqi_train.txt'
    test_data_file = 'zhengqi_test.txt'
    train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
    test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')

    return train_data, test_data


# 显示数据的基本信息
def show_basic_info(train_data, test_data):
    print(train_data.info())
    print(test_data.info())
    print(train_data.describe())
    print(test_data.describe())
    print(train_data.head())
    print(test_data.head())


# 找出数据中的异常值并删除
def handle_abnormal_feature(train_data, test_data):
    # 绘制各个特征的箱线图
    plt.figure(figsize=(18, 10))
    plt.boxplot(x=train_data.values, labels=train_data.columns)
    plt.hlines([-7.5, 7.5], 0, 40, colors='r')
    plt.show()

    # 由箱线图可知V9特征存在某些异常变量，删去
    # train_data = train_data[train_data['V9'] > -7.5]
    # test_data = test_data[test_data['V9'] > -7.5]

    # 绘制各个特征的KDE分布图
    fig = plt.figure(figsize=(30, 52))
    i = 1
    for col in test_data.columns:
        fig.add_subplot(5, 8, i)
        sns.kdeplot(train_data[col], color='Red', shade=True)
        sns.kdeplot(test_data[col], color='Blue', shade=True)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.legend(['train', 'test'])
        i += 1
    plt.show()

    # 存在异常的特征变量的KDE分布图
    drop_columns = ['V5', 'V9', 'V11', 'V17', 'V22', 'V27']
    plt.figure(figsize=(30, 52))
    i = 1
    for col in drop_columns:
        ax = plt.subplot(2, 3, i)
        ax = sns.kdeplot(train_data[col], color='Red', shade=True)
        ax = sns.kdeplot(test_data[col], color='Blue', shade=True)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.legend(['train', 'test'])
        i += 1
    plt.show()

    # 由KDE分布图可知V5,V9,V11,V17,V22,V27在训练数据和测试数据中的分布不一致，删去
    train_data = train_data.drop(drop_columns, axis=1)
    test_data = test_data.drop(drop_columns, axis=1)

    return train_data, test_data


# 对数据进行归一化处理
def normal_data(train_data, test_data):
    features = [col for col in train_data.columns if col not in ['target']]
    min_max_scaler = preprocessing.MinMaxScaler().fit(train_data[features])
    train_data_scaler = pd.DataFrame(min_max_scaler.transform(train_data[features]))
    test_data_scaler = pd.DataFrame(min_max_scaler.transform(test_data[features]))
    train_data_scaler.columns = features
    test_data_scaler.columns = features
    train_data_scaler['target'] = train_data['target']

    return train_data_scaler, test_data_scaler


# PCA降维，保留16个主成分
def dec_data(train_data_scaler, test_data_scaler):
    pca = PCA(n_components=16)
    new_train_data = pca.fit_transform(train_data_scaler.iloc[:, 0:-1])
    new_train_data = pd.DataFrame(new_train_data)
    new_train_data['target'] = train_data_scaler['target']
    new_test_data = pca.transform(test_data_scaler)
    new_test_data = pd.DataFrame(new_test_data)

    return new_train_data, new_test_data


# 将降维后的数据切分为训练数据和验证数据(80%,20%)
def split_data(new_train_data, new_test_data):
    new_train_data = new_train_data.fillna(0)
    train = new_train_data[new_test_data.columns]
    target = new_train_data['target']
    train_data, test_data, train_target, test_target = train_test_split(train, target, test_size=0.2, random_state=0)

    return train_data, test_data, train_target, test_target


# 尝试多种模型进行训练，观察各自的效果
def model_train(train_data, test_data, train_target, test_target):
    # 多元线性回归
    model = LinearRegression()
    model.fit(train_data, train_target)
    score = mean_squared_error(test_target, model.predict(test_data))
    print('LinearRegression:', score)

    # K近邻回归
    model = KNeighborsRegressor(n_neighbors=8)
    model.fit(train_data, train_target)
    score = mean_squared_error(test_target, model.predict(test_data))
    print('KNeighborsRegressor:', score)

    # 决策树回归
    model = DecisionTreeRegressor(random_state=0)
    model.fit(train_data, train_target)
    score = mean_squared_error(test_target, model.predict(test_data))
    print('DecisionTreeRegressor:', score)

    # 随机森林回归
    model = RandomForestRegressor(n_estimators=200)
    model.fit(train_data, train_target)
    score = mean_squared_error(test_target, model.predict(test_data))
    print('RandomForestRegressor:', score)

    # LGB模型回归
    model = lgb.LGBMModel(
        learning_rate=0.01,
        max_depth=-1,
        n_estimators=5000,
        boosting_type='gbdt',
        random_state=2020,
        objective='regression',
    )
    model.fit(train_data, train_target)
    score = mean_squared_error(test_target, model.predict(test_data))
    print('lightGbm:', score)


def main():
    train_data, test_data = read_data()
    show_basic_info(train_data, test_data)
    # 消除异常值
    train_data, test_data = handle_abnormal_feature(train_data, test_data)
    print(train_data.describe())
    print(test_data.describe())
    # 归一化
    train_data, test_data = normal_data(train_data, test_data)
    print(train_data.describe())
    print(test_data.describe())
    # PCA降维
    new_train_data, new_test_data = dec_data(train_data, test_data)
    print(new_train_data.describe())
    print(new_test_data.describe())

    # 模型训练，观察不同模型的效果
    train_data, test_data, train_target, test_target = \
        split_data(new_train_data, new_test_data)
    model_train(train_data, test_data, train_target, test_target)

    # 效果最好的是线性回归模型和LightGBM模型
    model = LinearRegression()
    model.fit(new_train_data[new_test_data.columns], new_train_data['target'])
    # model = lgb.LGBMModel(
    #     learning_rate=0.01,
    #     max_depth=-1,
    #     n_estimators=5000,
    #     boosting_type='gbdt',
    #     random_state=2020,
    #     objective='regression',
    # )
    # model.fit(new_train_data[new_test_data.columns], new_train_data['target'])

    # 写入预测结果
    res = model.predict(new_test_data)
    with open('res.txt', 'w') as f:
        for i in res:
            f.write(str(i) + '\n')
    f.close()


if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    main()
