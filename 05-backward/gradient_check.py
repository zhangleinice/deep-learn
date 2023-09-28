# 梯度确认
import os, sys
sys.path.append(os.pardir)
import numpy as np
from data.mnist import load_mnist
from two_layer_net import TwoLayerNet


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_numerical[key] - grad_backprop[key]))
    print(key+ ':' + str(diff))



# 数值微分和反向传播求出的梯度法非常小 ==>  反向传播结果正确
# 一旦你确保了网络的梯度计算正确性，就可以将其注释掉或删除，并专注于使用反向传播法来训练模型，以提高训练效率

# W1:5.846186062350221e-07
# b1:1.1344846629278852e-05
# W2:5.7642554137046795e-09
# b2:1.4035455174754575e-07