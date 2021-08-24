# Install
## Pytorch
PyTorch 默认使用 conda 来进行安装，例如

```conda install pytorch-cpu -c pytorch```
## MXNet
MXNet 更常用的是使用 pip。我们这里使用了 --pre 来安装 nightly 版本

```pip install --pre mxnet```
# 多维矩阵
## Pytorch
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
# MXNet
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
# 模型训练
我们使用一个多层感知机（MLP）来在 MINST 这个数据集上训练一个模型。我们将其分成 4 小块来方便对比。
## 读取数据
这里我们下载 MNIST 数据集并载入到内存，这样我们之后可以一个一个读取批量。
### PyTorch
```
import torch
from torchvision import datasets, transforms

train_data = torch.utils.data.DataLoader(
  datasets.MNIST(train=True, transform=transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.13,), (0.31,))])),
  batch_size=128, shuffle=True, num_workers=4)
  ```
### MXNet
```
from mxnet import gluon
from mxnet.gluon.data.vision import datasetes, transforms

train_data = gluon.data.DataLoader(
  datasets.MNIST(train=True).transform_first(transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(0.13, 0.31)])),
batch_size=128, shuffle=True, num_workers=4)
```








