import os
import sys
from data.mnist import load_mnist
from PIL import Image
import numpy as np

sys.path.append(os.pardir)


def img_show(img):
    pil = Image.fromarray(np.uint8(img))
    pil.show()


# flatten: 是否展开成一维数组
# normalize: 将图像正规化 0.0 ~ 1.0
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

# (60000, 784)
# (60000,)
# (10000, 784)
# (10000,)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
img_show(img)


# np.argmax()

# 创建一个示例数组
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5])

# 查找整个数组中的最大值的索引
max_index = np.argmax(arr)
print(max_index)  # 输出: 5

# 创建一个二维数组
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 在每列中查找最大值的索引
max_indices = np.argmax(matrix, axis=0)
print(max_indices)  # 输出: [2 2 2]

# 在每行中查找最大值的索引
max_indices = np.argmax(matrix, axis=1)
print(max_indices)  # 输出: [2 2 2]
