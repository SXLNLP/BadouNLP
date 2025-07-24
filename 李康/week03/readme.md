# week 2
## 任务

构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。

## 文件说明

RNNNet.py 模型定义

train.py 实现模型训练

predict.py 加载训练模型实现预测

utils 生成训练数据 格式转换函数等

## 网络配置

* 暂用pytorch自带的rnn做配置，接一层线性层转化所属类别

## 数据生成

随机生成5维字符，并人为添加一个a，

## 训练

迭代次数50次，训练数据1000个，batch size设置为5，Adam优化器，学习率设置0.001。验证集数据量1000个，记录并打印每轮次训练的平均loss以及验证准确率。训练完成后绘制每轮准确率、损失曲线，并存储训练模型。

训练结果记录：

epoch:49 eval acc:0.795

![](H:\1.ai_learning\2.project\4.week3\2.homework\pic\rnn_train.png)

## 模型预测

加载模型参数任意生成数据，并打印所属类别、准确率以及预测结果和真实结果可视化对比图。

以下是随机生成10组数据的结果：

![](H:\1.ai_learning\2.project\4.week3\2.homework\pic\result.png)

org_seq:['j', '[pad]', 'a', 'o', 'i'], org val:tensor([0., 0., 1., 0., 0.]), predict val:tensor([-4.3060, 10.9086,  6.4892, -9.9991, -1.9465]), org class:2, predict class:1
org_seq:['x', 'n', 'a', 'i', 'e'], org val:tensor([0., 0., 1., 0., 0.]), predict val:tensor([-9.6139, -4.6943,  8.9576, -0.8319,  5.3430]), org class:2, predict class:2
org_seq:['m', 'w', 'v', 'a', 'h'], org val:tensor([0., 0., 0., 1., 0.]), predict val:tensor([-1.0133, -9.1484, -5.1894,  8.9518,  4.6366]), org class:3, predict class:3
org_seq:['a', 'n', '[unk]', 'b', 'f'], org val:tensor([1., 0., 0., 0., 0.]), predict val:tensor([ 8.9959, -5.9705, -7.5369,  7.5809, -3.7504]), org class:0, predict class:0
org_seq:['e', 'h', 'l', 'f', 'a'], org val:tensor([0., 0., 0., 0., 1.]), predict val:tensor([-2.7230, -4.9554, -2.4115,  0.5683, 10.6423]), org class:4, predict class:4
org_seq:['x', 'j', 'h', 'f', 'a'], org val:tensor([0., 0., 0., 0., 1.]), predict val:tensor([-9.3271, -3.9709,  4.7453,  0.1090, 10.1007]), org class:4, predict class:4
org_seq:['a', 'f', 'o', 'n', 'r'], org val:tensor([1., 0., 0., 0., 0.]), predict val:tensor([13.1103,  4.9141, -5.3209, -7.6811, -4.0073]), org class:0, predict class:0
org_seq:['z', 'q', 'a', 's', 'o'], org val:tensor([0., 0., 1., 0., 0.]), predict val:tensor([-8.2142, -2.9496, 15.0827, -4.1318, -0.2025]), org class:2, predict class:2
org_seq:['a', 'm', 'e', '[unk]', 'f'], org val:tensor([1., 0., 0., 0., 0.]), predict val:tensor([  5.8224,  -4.8446,   5.7847, -12.5413,   6.5155]), org class:0, predict class:4
org_seq:['n', 'k', 'f', 'w', 'a'], org val:tensor([0., 0., 0., 0., 1.]), predict val:tensor([-6.0435, -2.2047,  6.5863, -9.8799, 13.2383]), org class:4, predict class:4
predict done==input:10 arrays, predict correct:8, predict incorrect:2
