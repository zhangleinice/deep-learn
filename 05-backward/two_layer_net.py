
# 两层神经网络
import os
import sys
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:
    # 初始化权重参数，以及网络层
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}

        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 顺序字典
        self.layers = OrderedDict()

        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    # 识别精度，准确率
    def accuracy(self, x, t):
        y = self.predict(x)
        # axis=1 在每行查找最大值的索引
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    # 数值微分求梯度（实现简单）
    def numerical_gradient(self, x, t):
        def loss_W(W): return self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # 反向传播求梯度（速度快）
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        # 反转顺序求导
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads


# # 初始化W, B参数，并计算梯度
# net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
# W1 = net.params['W1'].shape
# W2 = net.params['W2'].shape
# b1 = net.params['b1'].shape
# b2 = net.params['b2'].shape
# print('W1:', W1)
# print('W2:', W2)
# print('b1:', b1)
# print('b2:', b2)
# # W1: (784, 100)
# # W2: (100, 10)
# # b1: (100,)
# # b2: (10,)

# x = np.random.rand(100, 784)  # 输入100张图，每张图大小784

# t = np.random.rand(100, 10)
# # grads = net.numerical_gradient(x, t)
# grads = net.gradient(x, t)

# print(grads['W1'].shape)
# print(grads['W2'].shape)
# print(grads['b1'].shape)
# print(grads['b2'].shape)

# # (784, 100)
# # (100, 10)
# # (100,)
# # (10,)
