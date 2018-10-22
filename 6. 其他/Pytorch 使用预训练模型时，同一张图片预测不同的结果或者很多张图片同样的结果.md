

```python
from torchvision import datasets, models, transforms

alexnet_model = models.alexnet(pretrained=True)
alexnet_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

img = io.imread("cat.jpg")
img = transform(img)
img = img.unsqueeze(0)

prob = F.softmax(alexnet_model(image), dim=1)
pred = torch.argmax(prob, dim=1)
```

以上代码使用pytorch预训练的alexnet进行预测一张猫的图片，如果不加上`alexnet_model.eval()`则会导致每次运行预测的结果都不一样。
具体原因在[Pretrained resnet constant output](https://discuss.pytorch.org/t/pretrained-resnet-constant-output/2760)有解释：

因为`BatchNorm`和`Dropout`是随机的，会导致每次运行都不一致，所以在传递输入之前，需要切换至`model.eval()`进行预测。或者使用`model.train()`来进行训练。
