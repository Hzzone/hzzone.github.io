[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
Data-Driven Approach，数据驱动方法

## 参考资料
* [slides](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture02.pdf)
* [bilibili video](https://www.bilibili.com/video/av17204303/?p=4)
* [course notes](http://cs231n.github.io/classification/)

## 笔记
* 分类是指预先设定一张图片的语义（semantic）标签，例如 cat，由计算机预测。
* 图像由矩阵构成，当视角、光照、变形、遮挡，图片背景混乱（比如毛的纹理和背景相似），猫的年龄、毛发颜色不同，等条件变化时，图像是完全不一样的，这就需要我们的算法足够鲁棒（Robust，        对于聚类算法而言，鲁棒性意味着聚类结果不应受到模型中存在的数据扰动、噪声及离群点的太大影响。）
* high-end coded rules（硬编码规则）来识别动物，例如边缘信息，猫有耳朵、眼睛等，然后通过边缘、角度等信息组合在一起通过预先设置的规则进行分类。但是当识别其他类别时，也需要写规则。
* Data-Driven Approach，数据驱动的方法不通过预先设置的规则进行分类，而是通过手机大量数据，训练一个分类器，总结一个 model，总结对象的核心知识要素，最后识别新的图片。
* 所以分为两个过程，train 训练模型，predict 预测模型。

## 数据
超参数（hyperparameter），例如 K-Nearest-Neighbor 中的 K、距离度量等，不在训练中获得，而需要我们提前设置。
* 根据最高准确率选择超参（不是很好，例如 K=1 时，Nearest-Neighbor 永远都能分类的很好）
* 机器学习更关注的是如何在训练集以外的数据集（测试集、验证集）上表现的更好。如果只分为训练集和测试集，那我们训练的算法只是选一种在测试集上表现结果最好的，测试集不能代表所有的 unseen data，而算法在其他 unseen 的数据集上表现可能并不好。
* 将数据集分为三组，**训练集、验证集、测试集**，在训练集上训练，所以步骤完成之后，选择一组在验证集上效果最好的参数，最后在测试集上跑一边，然后论文中的数据是测试集上的实验结果。只有在最后才使用测试集，以保证算法在全新数据集上的效果是一样的。
<div align=center>
    <img src="https://hzzone.io/images/Screen%20Shot%202018-10-07%20at%204.03.24%20PM.png">
</div>

* 交叉验证（Cross Validation），通常在小数据集上使用，在深度学习很少用。例如将数据集分成 5 份（Five Cross Validation），进行五次操作，选择一份作为验证集，其他四份训练，最后可以得到那组超参最稳定。但是深度学习需要花大量时间训练，所以不常用。
<div align=center>
    <img src="https://hzzone.io/images/Screen%20Shot%202018-10-07%20at%204.48.38%20PM.png">
</div>

每个点代表一次结果，X 轴是 K 的值，Y 轴是准确度，线由均值连接，bar 是方差，然后准确度最高，方差最小的明显是 k ~= 7。

<div align=center>
    <img src="https://hzzone.io/images/Screen%20Shot%202018-10-07%20at%204.57.22%20PM.png">
</div>


* 测试集数据是否可以代表现实世界中的所有数据？

统计学假设数据都互相独立，符合同一概率分布。

当然也会有测试集代表性不佳，不能很好的反映真实世界。数据创建者，一次性用同样的方法收集大量数据，然后随机分成训练集、测试集。

有一种陷阱是刚开始收集的数据作为训练集，后面收集的是测试集，然后两者之间可能会存在偏差。但只要整个数据集的划分是随机的，就可以避免。



