scikit-learn 实用机器学习方法

## 大纲
* [数据集分为训练集和测试集](#数据集分为训练集和测试集)





## 数据集分为训练集和测试集

参考链接：[使用sklearn将数据集分为训练集和测试集](https://blog.csdn.net/sinat_29957455/article/details/79477940)
```python
from sklearn.model_selection import train_test_split
#测试集为30%，训练集为70%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=0)
```
