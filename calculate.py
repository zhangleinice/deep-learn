import numpy as np

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

# A行 和 B列相乘再求和
# 1*5 + 2*7 = 19
# 3*5 + 4*7 = 43

C = np.dot(A, B)
print(C)
# [19 22]
#  [43 50]]

# 必须保证A[1] 和B[0]的个数相等
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

B = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

# C的形状：A行，B列
C = np.dot(A, B)
print(C)
# [[22 28]
#  [49 64]]


# 神经网络的内积
# 使用np，可以一次性计算出神经网络的Y值；


X = np.array([1, 2])
W = np.array([
    [1, 3, 5],
    [2, 4, 6]
])
Y = np.dot(X, W)
print(Y)
# [ 5 11 17]
# y1 = x1*w1 + x2*w2
# ...
