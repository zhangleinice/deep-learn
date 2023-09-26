from data.mnist import load_mnist
import numpy as np
import pickle


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x/sum_exp_x
    return y


def init_network():
    with open("data/sample_weight.pkl", "rb") as f:
        # pickle.load() 序列化
        network = pickle.load(f)

    return network


def get_data():
    # normalize: 正规化
    _, (x_test, t_test) = load_mnist(
        flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

# 分批处理
batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):

    x_batch = x[i: i+batch_size]
    y_batch = predict(network, x_batch)
    print('y_batch', y_batch)

    # axis=1 在每行查找最大值的索引
    p = np.argmax(y_batch, axis=1)
    print('p', p)

    # 预测正确的个数总和
    accuracy_cnt += np.sum(p == t[i: i+batch_size])
    print('accuracy_cnt', accuracy_cnt)
    # accuracy_cnt 9352

# 表示有 93.52% 的数据被正确分类了
print("Accuracy:", float(accuracy_cnt) / len(x))

# Accuracy:0.9352
