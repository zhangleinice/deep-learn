# 反向传播推导由各个导数相乘

import numpy as np
from common.gradient import softmax, cross_entropy_error


# y = x > 0 ? x : 0;
# dx = dout > 0 ? dout: 0
class ReLU:
    def __init__(self):
        # 布尔数组
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = np.copy(x)
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx


# 加权和层
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        # 对应张量
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    # self.W.T 表示权重矩阵 self.W 的转置
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


# （y1 - t1, y2 - t2）
class SoftmaxWithLoss:
    def __init__(self):
        self.y = None  # 神经网络输出
        self.t = None  # 监督数据
        self.loss = None  # 损失

    def forward(self, x, t):
        self.t = t
        # logits ==> 概率
        self.y = softmax(x)
        # 真实值y, 与预测值t的误差（评估整个数据集或一个批次的损失）
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout = 1):
        bach_size = self.t.shape[0]
        dx = (self.y - self.t) / bach_size
        return dx
