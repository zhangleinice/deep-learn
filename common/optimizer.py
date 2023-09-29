
import numpy as np

# 缺点：梯度的方向没有指向最小值的方向
class SGD:

    def __init__(self, lr = 0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.items():
            params[key] = params[key] - self.lr * grads[key]

# 优点：是有助于克服梯度下降中的局部极小值问题，同时可以加速收敛
class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        # 初始化速度
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        # 更新参数
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] = params[key] + self.v[key]

