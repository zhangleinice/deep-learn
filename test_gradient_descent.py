# 导数 & 梯度 & 梯度下降
import numpy as np


# 导数
def numerical_diff(f, x):
    h = 1e-4
    # 中心差分
    return (f(x + h) - f(x - h)) / 2*h


# y = 0.01*x**2 + 0.1 *x
def function_1(x):
    return 0.01*x**2 + 0.1*x


# x = 5的导数
x5 = numerical_diff(function_1, 5)
print('x5', x5)  # x5 1.9999999999908982e-09

x10 = numerical_diff(function_1, 10)
print('x10', x10)  # x10 2.999999999986347e-09


# 梯度
def numerical_gradient(f, x):
    h = 1e-4
    # 创建一个和 x 同样形状的零数组来存储梯度
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x +h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x - h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        # 还原
        x[idx] = tmp_val

    return grad


# f(x0, x1) = x0 ** 2 + x1 ** 2
def function_2(x):
    return x[0]**2 + x[1]**2


x34 = numerical_gradient(function_2, np.array([3.0, 4.0]))
print(x34)  # [6. 8.]

x02 = numerical_gradient(function_2, np.array([0.0, 2.0]))
print(x02)  # [0. 4.]


# 梯度下降
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for _ in range(step_num):

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


# 梯度下降求最小值 f(x0, x1) = x0 ** 2 + x1 ** 2
init_x = np.array([-3.0, 4.0])
fmin = gradient_descent(function_2, init_x, 0.1, 100)
print(fmin)  # [-6.11110793e-10  8.14814391e-10]
