
# 激活函数 step & sigmoid & relu
import numpy as np
import matplotlib.pyplot as plt

# def step_func(x):
#     if (x > 0):
#         return 1
#     else:
#         return 0


# 支持np数组
def step_func(x):
    y = x > 0
    return y.astype(int)


# x = np.array([-1.0, 1.0, 1.0])
# print(x) #[-1.  1.  1.]
# y = x > 0
# print(y) #[False  True  True]
# y = y.astype(int)
# print(y) #[0 1 1]

# print(step_func(x))  # [0 1 1]


def sigmoid(x):
    return 1/(1 + np.exp(-x))

# x> 0? x: 0


def relu(x):
    return np.maximum(0, x)


x = np.arange(-5, 5, 0.1)
# y = step_func(x)
# y = sigmoid(x)
y = relu(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
