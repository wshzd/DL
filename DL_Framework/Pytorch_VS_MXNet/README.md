# Install
## Pytorch
PyTorch 默认使用 conda 来进行安装，例如

```conda install pytorch-cpu -c pytorch```
## MXNet
MXNet 更常用的是使用 pip。我们这里使用了 --pre 来安装 nightly 版本

```pip install --pre mxnet```
# 多维矩阵
## Pytorch
```import torch
x = torch.ones(5,3)
y = x + 1
print(y)
```













