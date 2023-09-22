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
