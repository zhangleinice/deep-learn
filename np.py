
import numpy as np

# x = np.array([1.0, 2.0, 3.0])
# y = np.array([2.0, 4.0, 6.0])

# print(x+y)

# a = [1, 2]
# print(type(a))

# m = np.array([
#     [56, 51],
#     [13, 19],
#     [0, 4]
# ])
# print(m > 15)
# print(m[m > 15])

# broadcast
a = np.array([
    [1, 2],
    [3, 4]
])

b = np.array([10, 20])

print(a.shape)
# (2, 2)

print(a * 10)
# [[10 20]
#  [30 40]]

print(a * b)
# [[10 40]
#  [30 80]]
