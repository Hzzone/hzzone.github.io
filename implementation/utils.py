import numpy as np
from sklearn import preprocessing

def read_iris(data_path):
    # 150*5
    # X: m*n, X^T*X: n*n
    iris_whole_data = np.loadtxt(data_path, delimiter=',', dtype='str')
    iris_label = iris_whole_data[:, 4]
    le = preprocessing.LabelEncoder()
    iris_label = le.fit_transform(iris_label)
    return iris_whole_data[:, :4].astype(np.float), iris_label


if __name__ == '__main__':
    iris_data, iris_label = read_iris('../data/iris.csv')
