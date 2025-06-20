import  torch
import torch.nn as nn
import numpy as np

"""
****************注意：示例代码中每一行代码都要搞懂****************
"""


#使用torch计算交叉熵
ce_loss = nn.CrossEntropyLoss()
#假设有3个样本，每个都在做3分类
pred = torch.FloatTensor([[0.3, 0.1, 0.3],
                          [0.9, 0.2, 0.9],
                          [0.5, 0.4, 0.2]])

#正确的类别分别为1,2,0
target = torch.LongTensor([1,2,0])
"""
[1,2,0]中三个元素分别对应着样本数据中第1、2、3个元素的第几个子元素的概率是1
其实是：
[
[0,1,0],   #表示[0.3, 0.1, 0.3]当中的标签是第2类
[0,0,1],   #表示[0.9, 0.2, 0.9]当中的标签是第3类
[1,0,0]   #表示[0.5, 0.4, 0.2]当中的标签是第1类
]
"""
loss = ce_loss(pred, target)
print("torch输出交叉熵：\n", loss)



"""
注意
np.exp()，如果传的是标量值x，那么就是e的x次幂；如果传的是矩阵，那么就是将矩阵中每个元素x进行e的x次幂
np.sum()，axis=1代表沿着列的反向求和，即沿着水平方向求和。axis=0代表沿着行的反向求和，即沿着竖直方向求和。keepdims=True代表保持输入参数的维度
"""
#实现softmax激活函数
def softmax(matrix):
    # print("np.exp(pred)：\n", np.exp(matrix))
    # print("np.sum(np.exp(matrix)：\n", np.sum(np.exp(matrix), axis=1, keepdims=True))
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=1, keepdims=True)

# print("np.exp(pred)：\n", np.exp(pred))
# print("np.sum(np.exp(matrix)：\n", np.sum(np.exp(pred), axis=1, keepdims=True))
# tt = softmax(pred.numpy())
# print("softmax计算后的值：\n", tt)
# print("pred.shape：\n", pred.shape)


"""
enumerate函数：
是Python的内置函数，用于在遍历可迭代对象时同步获取元素的索引和值，返回由(index, value)组成的元组迭代器。
"""
#将输入转化为onehot矩阵
def to_one_hot(target, shape):
    one_hot_target = np.zeros(shape)
    ##############
    for i, t in enumerate(target):
        one_hot_target[i][t] = 1
    return one_hot_target

# t1 = to_one_hot(target, pred.shape)
# print("enumerate(target)计算后的值：\n", list(enumerate(target)))
# print("to_one_hot计算后的值：\n", t1)


"""
给出一组3*3的矩阵，以及
"""
#手动实现交叉熵
def cross_entropy(pred, target):
    print("pred-old：\n", pred)  ## [[0.3 0.1 0.3] [0.9 0.2 0.9] [0.5 0.4 0.2]]

    batch_size, class_num = pred.shape
    print("pred.shape：\n", pred.shape)  ##(3, 3)

    """
        Softmax函数是一个常用的激活函数，主要用于将输入张量的每个元素转换为 0 到 1 之间的概率值，且这些概率值的总和为1。
            至于为什么这么做，就代表概率，数学原理层面还是没懂？？？？？？？？？？？？？？？？
    """
    pred = softmax(pred)
    print("pred-softmax：\n", pred)  ## [[0.3547696  0.2904608  0.3547696 ] [0.40054712 0.19890581 0.40054712] [0.37797815 0.34200877 0.2800131 ]]

    """
        通过自定义的to_one_hot方法，还原了torch.LongTensor方法
    """
    print("target：\n", target)  ## [1 2 0]
    target = to_one_hot(target, pred.shape)
    print("target-to_one_hot：\n", target) ## [[0. 1. 0.][0. 0. 1.] [1. 0. 0.]]

    """
        使用numpy.sum方法，输入参数必须是numpy列表类型，求和的结果依然是numpy列表类型
        使用torch.sum方法，输入参数必须是tensor张量类型，求和的结果依然是tensor张量类型
    """
    entropy = - np.sum(target * np.log(pred), axis=1)
    print("entropy：\n", entropy)  ## [1.23628664 0.91492391 0.97291887]

    """
        通过sum(entropy)，计算了3组样本的交叉熵的综合
    """
    print("sum(entropy)：\n", sum(entropy))  ## 3.1241294145584106

    """
        除以了batch_size，计算了三组样本的平均交叉熵，作为这一批样本的交叉熵
    """
    return sum(entropy) / batch_size  ## 1.0413764715194702



print("手动实现交叉熵：\n", cross_entropy(pred.numpy(), target.numpy()))


# print("以e为底，2.7的对数", np.log(2.7))  ##0.9932517730102834
# print("以10为底，2.7的对数", np.log10(2.7))  ##0.43136376415898736