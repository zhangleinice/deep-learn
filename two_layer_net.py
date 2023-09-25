

import numpy as np
from common import sigmoid, softmax, cross_entropy_error, numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        #  axis=1 在每行查找最大值的索引
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    # 梯度计算
    def numerical_gradient(self, x, t):
        def loss_W(W): return self.loss(x, t)
        grads = {}

        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
W1 = net.params['W1'].shape
W2 = net.params['W2'].shape
b1 = net.params['b1'].shape
b2 = net.params['b2'].shape
print('W1:', W1)
print('W2:', W2)
print('b1:', b1)
print('b2:', b2)
# W1: (784, 100)
# W2: (100, 10)
# b1: (100,)
# b2: (10,)

x = np.random.rand(100, 784)  # 输入100张图，每张图大小784
y = net.predict(x)

t = np.random.rand(100, 10)
grads = net.numerical_gradient(x, t)

print(grads['W1'].shape)
print(grads['W2'].shape)
print(grads['b1'].shape)
print(grads['b2'].shape)

# (784, 100)
# (100, 10)
# (100,)
# (10,)
