# deep-learn

## 感知机
    1. AND，NAND， OR具有相同的感知机，只是权重和偏置不一样
    2. XOR 多层感知机


## 神经网络的学习
1.  TwoLayerNet类设计  
         \_\_init\_\_: : 初始化权重参数W,b，神经网络层layers（顺序字典）  
           predict: 顺序遍历layers, 前向传播  
              loss: 交叉熵作为损失函数  
          accuracy: 精确度计算  
numerical_gradient: 数值微分计算梯度  
          gradient: 反向传播，逆向遍历layers，计算梯度  

2.  随机梯度下降  
    随机选部分数据，使用梯度下降更新参数  

3.  监视训练过程  
    记录学习过程  
    计算每一代的识别精度  
    查看损失函数曲线；以及训练训练数据和测试数据的精确度图像，是否过拟合      
