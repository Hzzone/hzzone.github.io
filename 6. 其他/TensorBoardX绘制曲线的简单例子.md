[tensorboardX](https://github.com/lanpa/tensorboardX)是pytorch集成tensorboard可视化训练过程的工具。
### 绘制loss曲线的方法如下
1. 单条曲线使用`writer.add_scalar`，一条曲线对应一个name
2. 多条曲线使用`writer.add_scalars`，每个表格对应一个name，其中每条曲线也对应一个name

### Example
```python
from tensorboardX import SummaryWriter
import random
import time

writer = SummaryWriter()

for i in range(1000):
    for x in ['train', 'val']:
        writer.add_scalars("loss", {
            x: random.random()
        }, i)
    writer.add_scalar("acc", random.random(), i)
    time.sleep(1)

writer.close()
```
1. 单条曲线
![](http://tuchuang-1252747889.cosgz.myqcloud.com/2018-08-31-FireShot%20Capture%201%20-%20TensorBoard%20-%20http___127.0.0.1_6006_-scalars%26_showDownloadLinks-true.png)
2. 多条曲线
![](http://tuchuang-1252747889.cosgz.myqcloud.com/2018-08-31-FireShot%20Capture%202%20-%20TensorBoard%20-%20http___127.0.0.1_6006_-scalars%26_showDownloadLinks-true.png)