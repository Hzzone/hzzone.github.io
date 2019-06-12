### 参考

* [正则化](https://hit-scir.gitbooks.io/neural-networks-and-deep-learning-zh_cn/content/chap3/c3s5ss1.html)

### Weight Regularization（正则化）

Weight Regularization 在 cs231n 的 [Loss Functions and Optimization ](http://cs231n.stanford.edu/syllabus.html) 提及。我觉得这篇文章 [正则化](https://hit-scir.gitbooks.io/neural-networks-and-deep-learning-zh_cn/content/chap3/c3s5ss1.html) 写得很详细，可以参考一下，不过其中应该有个错误正则项应该是不需要除 $n$ 的。

下面是 cs231n 涉及到正则化的内容，非常直观，我就不再多写了:

![](https://tuchuang-1252747889.cosgz.myqcloud.com/2018-12-02-%E6%9C%AA%E5%91%BD%E5%90%8D%E6%8B%BC%E5%9B%BE%20-1-.jpg)

首先 Weight Regularization 是解决过拟合的一种方法，提高模型泛化能力，其他的还有 Dropout、Batch Norm 等。

Weight Regularization 起作用主要是约束模型复杂度，获得更简单的权重。

* cs231n 的例子

假设 $W$ 是最优解，$2W$ 的结果也一样。上图中 $w _ { 1 } ^ { T } x = w _ { 2 } ^ { T } x = 1$，$w_2$ 的 Frobenius 范数更小，所以 $w_2$ 更简单。区别在于 L2 范数将权重跟趋向于均匀分布（展开），而不是极端分布。

* [解析深度学习——卷积神经网络原理与视觉实践](http://lamda.nju.edu.cn/weixs/book/CNN_book.html)

>如图，如果将模型原始的假设空间比做“天空”， 那么天空中自由飞翔的“鸟”就是模型可能收敛到的一个个最优解。 在施加了模型正则化后，就好比将原假设空间（“天空”）缩小到一定的空间范围（“笼子”），这样一来，可能得到的最优解（“鸟”）能搜寻的假设空间也变得相对有限。有限空间自然对应复杂度不太高的模型，也自然对应了有限的模型表达能力，这就是“正则化能有效防止模型过拟合”的一种直观解释。许多浅层学习器（如支持向量机等）为了提高泛化性往往都要依赖模型正则 化，深度学习更应如此。深度网络模型相比浅层学习器巨大的多的模型复杂度 是把更锋利的双刃剑：保证模型更强大表示能力的同时也使模型蕴藏着更巨大的过拟合风险。深度模型的正则化可以说是整个深度模型搭建的最后一步，更是不可缺少的重要一步。

![](https://tuchuang-1252747889.cosgz.myqcloud.com/2018-12-03-%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202018-12-03%20%E4%B8%8B%E5%8D%889.50.54.png)

* Deep Learning 书上的例子

为什么不对偏置正则化:
>在探究不同范数的正则化表现之前，我们需要说明一下，在神经网络中，参数包括每一层仿射变换的权重和偏置，我们通常只对权重做惩罚而不对偏置做正则惩罚。 精确拟合偏置所需的数据通常比拟合权重少得多。每个权重会指定两个变量如何相互作用。我们需要在各种条件下观察这两个变量才能良好地拟合权重。而每个偏置仅控制一个单变量。这意味着，我们不对其进行正则化也不会导致太大的方差。另外，正则化偏置参数可能会导致明显的欠拟合。因此，我们使用向量 w 表示所有应受范 数惩罚影响的权重，而向量 θ 表示所有参数 (包括 w 和无需正则化的参数)。在神经网络的情况下，有时希望对网络的每一层使用单独的惩罚，并分配不同的 α 系数。寻找合适的多个超参数的代价很大，因此为了减少搜索空间，我们会在所有层使用相同的权重衰减。

书中在 **第七章 深度学习中的正则化** 对正则化的证明、作用有很深的探讨，比上面的两个介绍的多。而且证明的很漂亮，强烈推荐。

#### L2 正则化

$L^2$ 参数范数惩罚又被叫做权重衰减（weight decay），在其他学术圈，也被称为岭回归或 Tikhonov 正则。

$L^2$ 正则化的公式为:

$$
\tilde { J } (  w  ;  X  ,  y  ) = \frac { \alpha } { 2 } \| \boldsymbol { w } \| _ { 2 } ^ { 2 }+ J (  w  ;  X ,  y  )
$$

$\alpha$ 控制正则项大小，较大的 $\alpha$ 取值将较大程度约束模型复杂度；反之易然。

在原来的损失函数基础上加上了 $\Omega ( \boldsymbol { \theta } ) = \frac { 1 } { 2 } \| \boldsymbol { w } \| _ { 2 } ^ { 2 }$ 的正则项。

对 $w$ 进行求导:

$$
\nabla _ { w } \tilde { J } ( w  ;  X  , y  ) = \alpha  w  + \nabla _ { w } J (  w  ;  X  ,  y  )
$$

更新权重时的公式变为:

$$
\boldsymbol { w } \leftarrow  w  - \epsilon \left( \alpha w  + \nabla _ { w } J (  w  ; X  ,y  ) \right)
$$

$$
 w  \leftarrow ( 1 - \epsilon \alpha )  w  - \epsilon \nabla _ { w } J (  w  ; X  ,  y  )
$$

在每步执行通常的梯度更新之前先收缩权重向量（将权重向量乘以一个常数因子）。

再贴一些来自 Deep Learning 书上的说明:

> 只有在显著减小目标函数方向上的参数会保留得相对完好。在无助于目标函 数减小的方向（对应 Hessian 矩阵较小的特征值）上改变参数不会显著增加梯度。这种不重要方向对应的分量会在训练过程中因正则化而衰减掉。

> 我们可以看到，L2 正则化能让学习算法 ‘‘感知’’ 到具有较高方差的输入 x，因此与输出目标的协方差较小（相对增加方差）的特征的权重将会收缩。

有些地方我也没看懂，难过。

### L1 正则化

L1 正则项为:

$$
\Omega (  \theta  ) = \|  w  \| _ { 1 } = \sum _ { i } \left| w _ { i } \right|
$$

整体代价函数:

$$
\tilde { J } (  w  ;  X  ,  y  ) = \alpha \|  w  \| _ { 1 } + J (  w  ;  X  ,  y  )
$$

对应的梯度为:

$$
\nabla _ { w } \tilde { J } (  w  ;  X  ,  y  ) = \alpha \operatorname { sign } (  w  ) + \nabla _ { w } J (  w  ;  X  ,  y  )
$$

sign 函数大于 0 为 1，小于 0 为 -1，等于 0 为 0。

L1 与 L2 的不同之处:

>我们立刻发现 L1 的正则化效果与 L2 大不一样。具体来说，我们可以看到正则化对梯度的影响不再是线性地缩放每个 $w_i$；而是添加了一项与 $sign(w_i)$ 同号的常数。

>相比 L2 正则化，L1 正则化会产生更稀疏（sparse）的解。此处稀疏性指的是最优值中的一些参数为 0。

> 稀疏化的结果使优化后的参数一部分为 0，另一部分为非零实值。非零实值的那部分参数可起到选择重要参数或特征维度的作用，同时可起到去除噪声的效果。

#### Elastic 正则化

联合使用 L1、L2 正则化，正则项为:

$$
\alpha _ { 1 } \| \omega \| _ { 1 } + \alpha _ { 2 } \| \alpha \| _ { 2 } ^ { 2 }
$$

### 正则化实现

我前面实现了线性层、交叉熵等，我直接复制不需要改的代码，然后修改线性层实现包括 L1、L2 正则化。

再对比一下三个的效果。

**复制不需要改的代码，保存在最后一个单元格。**

原先的实现没有正则化，在这里直接修改，关于梯度在上面已经求了。


```python
def l1_regularization(W, alpha):
    return alpha*np.sign(W)

def l2_regularization(W, alpha):
    return alpha*W

def no_regularization(W, alpha):
    return np.zeros_like(W)

regularize = {
    0: no_regularization,
    1: l1_regularization,
    2: l2_regularization
}
class Linear(object):
    def __init__(self, D_in, D_out, regularization=0, alpha=0):
        self.weight = np.random.randn(D_in, D_out).astype(np.float32)*0.01
        self.bias = np.zeros((1, D_out), dtype=np.float32)
        self.regularization = regularization
        self.alpha = alpha
        
    def forward(self, input):
        self.data = input
        return np.dot(self.data, self.weight)+self.bias
        
    def backward(self, top_grad, lr):
        self.grad = np.dot(top_grad, self.weight.T).astype(np.float32)
        grad_w = np.dot(self.data.T, top_grad)
        # 加上正则项求导
        grad_w += regularize[self.regularization](self.weight, self.alpha)
        # 更新参数
        self.weight -= lr*grad_w
        self.bias -= lr*np.mean(top_grad, axis=0)
```

更新了一下线性层如果加上正则化项之后的反向传播的关于权重 $w$ 的梯度，bias 没有正则项。

我在写一个通用的训练函数，测试一下这三种情况的结果有什么区别。


```python
from tqdm import tqdm_notebook
import copy

batch_size = 120
# 读取并归一化数据，不归一化会导致 nan
test_data = ((read_mnist('../data/mnist/t10k-images.idx3-ubyte').reshape((-1, 784))-127.0)/255.0).astype(np.float32)
train_data = ((read_mnist('../data/mnist/train-images.idx3-ubyte').reshape((-1, 784))-127.0)/255.0).astype(np.float32)
# 独热编码标签
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoder.fit(np.arange(10).reshape((-1, 1)))
train_labels = encoder.transform(read_mnist('../data/mnist/train-labels.idx1-ubyte').reshape((-1, 1))).toarray().astype(np.float32)
test_labels = encoder.transform(read_mnist('../data/mnist/t10k-labels.idx1-ubyte').reshape((-1, 1))).toarray().astype(np.float32)
train_dataloader = Dataloader(train_data, train_labels, batch_size, shuffle=True)
test_dataloader = Dataloader(test_data, test_labels, batch_size, shuffle=False)

# net 应该是一个一层的线性网络
def train_and_test(loss_layer, net, scheduler, max_iter, train_dataloader, test_dataloader):
    test_loss_list, train_loss_list, train_acc_list, test_acc_list = [], [], [], []
    best_net = None
    # 最高准确度，和对应权重
    best_acc = -float('inf')
    for epoch in tqdm_notebook(range(max_iter)):
        # 训练
        correct = 0
        total_loss = 0
        for data, labels in train_dataloader:
            # 前向输出概率
            train_pred = net.forward(data)

            # 计算准确度
            pred_labels = np.argmax(train_pred, axis=1)
            real_labels = np.argmax(labels, axis=1)
            correct += np.sum(pred_labels==real_labels)

            # 前向输出损失
            loss = loss_layer.forward(train_pred, labels)
            total_loss += loss*data.shape[0]
            

            # 反向更新参数
            loss_layer.backward()
            net.backward(loss_layer.grad, scheduler.get_lr())
            
        total_loss /= len(train_dataloader)
        if net.regularization==0:
            reg_loss = 0
        elif net.regularization==1:
            reg_loss = np.sum(net.weight)*net.alpha
        else:
            reg_loss = np.sqrt(np.sum(np.square(net.weight)))*net.alpha/2
        total_loss += reg_loss
        
        acc = correct/len(train_dataloader)
        train_acc_list.append(acc)
        train_loss_list.append(total_loss)
        scheduler.step()
        
        # 测试
        correct = 0
        total_loss = 0
        for data, labels in test_dataloader:
            # 前向输出概率
            test_pred = net.forward(data)

            # 前向输出损失
            loss = loss_layer.forward(test_pred, labels)
            total_loss += loss*data.shape[0]

            # 计算准确度
            pred_labels = np.argmax(test_pred, axis=1)
            real_labels = np.argmax(labels, axis=1)
            correct += np.sum(pred_labels==real_labels)
            
        total_loss /= len(test_dataloader)
        
        # 正则项损失因为没有更新参数所以不变
        total_loss += reg_loss
        
        acc = correct/len(test_dataloader)
        test_acc_list.append(acc)
        test_loss_list.append(total_loss)

        if acc > best_acc: 
            best_acc = acc
            best_net = copy.deepcopy(net)
    return test_loss_list, train_loss_list, train_acc_list, test_acc_list, best_net
```

初始化各项参数:


```python
# 损失层
loss_layer = CrossEntropyLossLayer()
# 输入输出维度
D, C = 784, 10
np.random.seed(1) # 固定随机生成的权重
```

开始训练:


```python
# 最大迭代次数和步长
max_iter = 120
step_size = 50
# 学习率
lr = 0.1
# 学习率衰减
scheduler = lr_scheduler(lr, step_size)
linear_classifer_0 = Linear(D, C)
test_loss_list0, train_loss_list0, train_acc_list0, test_acc_list0, best_net0 = train_and_test(loss_layer, linear_classifer_0, scheduler, max_iter, train_dataloader, test_dataloader)
```




    



```python
def show(max_iter, train_loss_list, test_loss_list, train_acc_list, test_acc_list):
    plt.subplot(2, 1, 1)
    plt.title('loss')
    plt.plot(range(max_iter), train_loss_list, label='train_loss')
    plt.plot(range(max_iter), test_loss_list, label='test_loss')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title('accuracy')
    plt.plot(range(max_iter), train_acc_list, label='train_acc')
    plt.plot(range(max_iter), test_acc_list, label='test_acc')
    plt.legend()
    plt.subplots_adjust(hspace=0.5)
    
show(max_iter, train_loss_list0, test_loss_list0, train_acc_list0, test_acc_list0)
```


![](https://tuchuang-1252747889.cosgz.myqcloud.com/2018-12-05-output_50_0.png)



```python
# 最大迭代次数和步长
max_iter = 120
step_size = 50
# 学习率
lr = 0.1
# 学习率衰减
scheduler = lr_scheduler(lr, step_size)
linear_classifer_1 = Linear(D, C, regularization=1, alpha=1e-4)
test_loss_list1, train_loss_list1, train_acc_list1, test_acc_list1, best_net1 = train_and_test(loss_layer, linear_classifer_1, scheduler, max_iter, train_dataloader, test_dataloader)
```


    HBox(children=(IntProgress(value=0, max=120), HTML(value='')))


    



```python
show(max_iter, train_loss_list1, test_loss_list1, train_acc_list1, test_acc_list1)
```


![](https://tuchuang-1252747889.cosgz.myqcloud.com/2018-12-05-output_52_0.png)



```python
# 最大迭代次数和步长
max_iter = 120
step_size = 50
# 学习率
lr = 0.1
# 学习率衰减
scheduler = lr_scheduler(lr, step_size)
linear_classifer_2 = Linear(D, C, regularization=2, alpha=1e-3)
test_loss_list2, train_loss_list2, train_acc_list2, test_acc_list2, best_net2 = train_and_test(loss_layer, linear_classifer_2, scheduler, max_iter, train_dataloader, test_dataloader)
```


    HBox(children=(IntProgress(value=0, max=120), HTML(value='')))


    



```python
show(max_iter, train_loss_list2, test_loss_list2, train_acc_list2, test_acc_list2)
```


![](https://tuchuang-1252747889.cosgz.myqcloud.com/2018-12-05-output_54_0.png)


可视化无正则、L1 正则、L2 正则的权重差别:


```python
import numpy as np
import matplotlib.pyplot as plt

plt.hist(best_net0.weight.ravel(), bins=np.arange(-1, 1, 0.01), label='no regularization')
plt.hist(best_net1.weight.ravel(), bins=np.arange(-1, 1, 0.01), label='l1 regularization')
plt.hist(best_net2.weight.ravel(), bins=np.arange(-1, 1, 0.01), label='l2 regularization')
plt.legend()
plt.show()
```


![](https://tuchuang-1252747889.cosgz.myqcloud.com/2018-12-05-output_56_0.png)


在可视化一下三者之间的收敛速度:

**从以上的结果可以得出几个结论:**

* 相对于无正则化之后的权重方差更小。
* 和 L2 相比，L1 使权重更稀疏，看他的 0 更突出。
* 同样的参数下 L2 的约束能力比 L1 要强，所以需要注意一下正则参数的大小不要太大，否则不能收敛（欠拟合）。
* test accuracy 和 train accuracy 之间的 gap 变小（正则化的意义）。


```python
np.std(best_net0.weight), np.std(best_net1.weight), np.std(best_net2.weight)
```




    (0.18500698, 0.14412738, 0.11477291)



测试集上的准确度也没有太大区别，一点小差距调整下步长和迭代次数就可以 work 了。


```python
np.max(test_acc_list0), np.max(test_acc_list1), np.max(test_acc_list2)
```




    (0.9257, 0.9232, 0.9218)




```python

```

**最后总结一下，为了防止过拟合，减小 test accuracy 和 train accuracy 之间的 gap，非常需要权重衰减，获得更简单的权重。**

**一下是需要的代码**


```python
import numpy as np
import struct
def softmax(input):
    exp_value = np.exp(input) #首先计算指数
    output = exp_value/np.sum(exp_value, axis=1)[:, np.newaxis] # 然后按行标准化
    return output

class CrossEntropyLossLayer():
    def __init__(self):
        pass
    
    def forward(self, input, labels):
        # 做一些防止误用的措施，输入数据必须是二维的，且标签和数据必须维度一致
        assert len(input.shape)==2, '输入的数据必须是一个二维矩阵'
        assert len(labels.shape)==2, '输入的标签必须是独热编码'
        assert labels.shape==input.shape, '数据和标签数量必须一致'
        self.data = input
        self.labels = labels
        self.prob = np.clip(softmax(input), 1e-9, 1.0) #在取对数时不能为 0，所以用极小数代替 0
        loss = -np.sum(np.multiply(self.labels, np.log(self.prob)))/self.labels.shape[0]
        return loss
    
    def backward(self):
        self.grad = (self.prob - self.labels)/self.labels.shape[0] # 根据公式计算梯度

class Dataloader(object):
    def __init__(self, data, labels, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.labels = labels
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __iter__(self):
        datasize = self.data.shape[0]
        data_seq = np.arange(datasize)
        if self.shuffle:
            np.random.shuffle(data_seq)
        interval_list = np.append(np.arange(0, datasize, self.batch_size), datasize)
        for index in range(interval_list.shape[0]-1):
            s = data_seq[interval_list[index]:interval_list[index+1]]
            yield self.data[s], self.labels[s]
    
    def __len__(self):
        return self.data.shape[0]
class lr_scheduler(object):
    def __init__(self, base_lr, step_size, deacy_factor=0.1):
        self.base_lr = base_lr # 最初的学习率
        self.deacy_factor = deacy_factor # 学习率衰减因子
        self.step_count = 0 # 当前的迭代次数
        self.lr = base_lr # 当前学习率
        self.step_size = step_size # 步长
        
    def step(self, step_count=1): # 默认 1 次
        self.step_count += step_count
    
    def get_lr(self):
        # 根据公式 12 实现
        self.lr = self.base_lr*(self.deacy_factor**(self.step_count//self.step_size)) # 实现上面的公式
        return self.lr

def read_mnist(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
```


```python

```
