# Step1：Install
## Pytorch
PyTorch 默认使用 conda 来进行安装，例如

```conda install pytorch-cpu -c pytorch```
## MXNet
MXNet 更常用的是使用 pip。我们这里使用了 --pre 来安装 nightly 版本

```pip install --pre mxnet```
# Step2：Operation
## 多维矩阵
### Pytorch
```
import torch
x = torch.ones(5,3)
y = x + 1
print(y)
```
```
2  2  2
2  2  2
2  2  2
2  2  2
2  2  2
[torch.FloatTensor of size 5x3]
```
### MXNet
```
from mxnet import nd
x = nd.ones((5,3))
y = x + 1
print(y)
```
```
[[2. 2. 2.]
[2. 2. 2.]
[2. 2. 2.]
[2. 2. 2.]
[2. 2. 2.]]
<NDArray 5x3 @cpu(0)>
```
这里主要的区别是 MXNet 的形状传入参数跟 NumPy 一样需要用括号括起来。
## 模型训练
我们使用一个多层感知机（MLP）来在 MINST 这个数据集上训练一个模型。我们将其分成 4 小块来方便对比。
### 读取数据
这里我们下载 MNIST 数据集并载入到内存，这样我们之后可以一个一个读取批量。
#### PyTorch
```
import torch
from torchvision import datasets, transforms

train_data = torch.utils.data.DataLoader(
  datasets.MNIST(train=True, transform=transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.13,), (0.31,))])),
  batch_size=128, shuffle=True, num_workers=4)
  ```
#### MXNet
```
from mxnet import gluon
from mxnet.gluon.data.vision import datasetes, transforms

train_data = gluon.data.DataLoader(
  datasets.MNIST(train=True).transform_first(transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(0.13, 0.31)])),
batch_size=128, shuffle=True, num_workers=4)
```
这里的主要区别是 MXNet 使用 transform_first 来表明数据变化是作用在读到的批量的第一个元素，既 MNIST 图片，而不是第二个标号元素。
### 定义模型
下面我们定义一个只有一个单隐层的 MLP
#### PyTorch
```
from torch import nn

net = nn.Sequential(
  nn.Linear(28*28, 256),
  nn.ReLU(),
  nn.Linear(256, 10)
)
```
#### MXNet
```
from mxnet.gluon import nn

net = nn.Sequential()
with net.name_scope():
  net.add(
      nn.Dense(256, activation='relu'),
      nn.Dense(10)
  )
net.initialize()
```
我们使用了 Sequential 容器来把层串起来构造神经网络。这里MXNet跟PyTorch的主要区别是：
不需要指定输入大小，这个系统会在后面自动推理得到
全连接和卷积层可以指定激活函数
需要创建一个 name_scope 的域来给每一层附上一个独一无二的名字，这个在之后读写模型时需要
我们需要显示调用模型初始化函数。
大家知道 Sequential 下只能神经网络只能逐一执行每个层。PyTorch可以继承 nn.Module 来自定义 forward 如何执行。同样，MXNet 可以继承 nn.Block 来达到类似的效果。

















