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

>不需要指定输入大小，这个系统会在后面自动推理得到

>全连接和卷积层可以指定激活函数

>需要创建一个 name_scope 的域来给每一层附上一个独一无二的名字，这个在之后读写模型时需要我们需要显示调用模型初始化函数。

>大家知道 Sequential 下只能神经网络只能逐一执行每个层。PyTorch可以继承 nn.Module 来自定义 forward 如何执行。同样，MXNet 可以继承 nn.Block 来达到类似的效果。
### 损失函数和优化算法
#### PyTorch
```
loss_fn = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```
#### MXNet
```
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                      'sgd', {'learning_rate': 0.1})
```
这里我们使用交叉熵函数和最简单随机梯度下降并使用固定学习率 0.1
## 训练
最后我们实现训练算法，并附上了输出结果。注意到每次我们会使用不同的权重和数据读取顺序，所以每次结果可能不一样。
#### PyTorch
```
from time import time
for epoch in range(5):
  total_loss = .0
  tic = time()
  for X, y in train_data:
      X, y = torch.autograd.Variable(X), torch.autograd.Variable(y)
      trainer.zero_grad()
      loss = loss_fn(net(X.view(-1, 28*28)), y)
      loss.backward()
      trainer.step()
      total_loss += loss.mean()
  print('epoch %d, avg loss %.4f, time %.2f' % (
      epoch, total_loss/len(train_data), time()-tic))
```
```
epoch 0, avg loss 0.3251, time 3.71
epoch 1, avg loss 0.1509, time 4.05
epoch 2, avg loss 0.1057, time 4.07
epoch 3, avg loss 0.0820, time 3.70
epoch 4, avg loss 0.0666, time 3.63
```
#### MXNet
```
from time import time
for epoch in range(5):
  total_loss = .0
  tic = time()
  for X, y in train_data:
      with mx.autograd.record():
        loss = loss_fn(net(X.flatten()), y)
      loss.backward()
      trainer.step(batch_size=128)
      total_loss += loss.mean().asscalar()
  print('epoch %d, avg loss %.4f, time %.2f' % (
      epoch, total_loss/len(train_data), time()-tic))
```
```
epoch 0, avg loss 0.3162, time 1.59
epoch 1, avg loss 0.1503, time 1.49
epoch 2, avg loss 0.1073, time 1.46
epoch 3, avg loss 0.0830, time 1.48
epoch 4, avg loss 0.0674, time 1.75
```
MXNet 跟 PyTorch 的不同主要在下面这几点：

不需要将输入放进 Variable， 但需要将计算放在 mx.autograd.record() 里使得后面可以对其求导

不需要每次梯度清 0，因为新梯度是写进去，而不是累加

step 的时候 MXNet 需要给定批量大小

需要调用 asscalar() 来将多维数组变成标量。

这个样例里 MXNet 比 PyTorch 快两倍。当然大家对待这样的比较要谨慎。
