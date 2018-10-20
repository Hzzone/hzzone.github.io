[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)

KNN (K-Nearest-Neighbor Classifier)，K最邻近算法

## 参考资料
* [slides](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture02.pdf)
* [bilibili video](https://www.bilibili.com/video/av17204303/?p=5)
* [course notes](http://cs231n.github.io/classification/)

## 笔记

### KNN
[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) 由 60K 张 32x32 的图片构成，分为 10 类，每类 6K，训练集 50K，测试集 10K。
<div align=center>
   <img src="https://tuchuang-1252747889.cosgz.myqcloud.com/2018-10-20-131318.png" />
</div>

* Nearest Neighbor

最邻近分类很少被使用，但是可以用来理解图像分类问题。

最邻近分类很简单，一张测试图片和每一张训练图片进行比较，然后输出的是和测试图片最接近的那张训练图片的标签。

图片的相似程度则使用 L1 或 L2 距离来衡量，具体方法是像素间的距离。例如计算 L1 距离，矩阵对应位置相减求绝对值再相加:
<div align=center>
   <img src="https://tuchuang-1252747889.cosgz.myqcloud.com/2018-10-20-Screen%20Shot%202018-10-20%20at%209.13.53%20PM.png" />
</div>

