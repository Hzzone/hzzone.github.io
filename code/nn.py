import numpy as np
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


class Linear(object):
    def __init__(self, D_in, D_out):
        self.weight = np.random.randn(D_in, D_out).astype(np.float32)*0.01
        self.bias = np.zeros((1, D_out), dtype=np.float32)
        
    def forward(self, input):
        self.data = input
        return np.dot(self.data, self.weight)+self.bias
        
    def backward(self, top_grad, lr):
        self.grad = np.dot(top_grad, self.weight.T).astype(np.float32)
        # 更新参数
        self.weight -= lr*np.dot(self.data.T, top_grad)
        self.bias -= lr*np.mean(top_grad, axis=0)
        
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