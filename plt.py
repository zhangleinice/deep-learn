
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread


# 创建一个 x 值的数组，范围是从 0 到 6，以 0.1 为步长
x = np.arange(0, 6, 0.1)
y = np.sin(x)
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制
# plt.plot(x, y1, label="sin")
# plt.plot(x, y2, label="cos", linestyle="--")

# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('sin & cos')
# # 显示图例，以便区分 sin 和 cos 曲线
# plt.legend()
# plt.show()

img = imread('data/test.png')
plt.imshow(img)
plt.show()
