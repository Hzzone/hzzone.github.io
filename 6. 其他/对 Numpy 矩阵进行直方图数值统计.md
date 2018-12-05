
## 对 Numpy 矩阵进行直方图数值统计

说实话对矩阵进行可视化在深度学习中需求很大，例如可视化权重分布。

[np.ravel](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ravel.html) 将矩阵展开，`plt.hist` 绘制直方图。


```python
import numpy as np
import matplotlib.pyplot as plt

a = np.random.randn(1000, 1000)
plt.hist(a.ravel(), bins=np.arange(np.min(a), np.max(a), 0.01))
plt.show()
```


![](https://tuchuang-1252747889.cosgz.myqcloud.com/2018-12-05-output_3_0.png)
