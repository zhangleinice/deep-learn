
# 乘法层
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y  # 翻转 x, y
        dy = dout * self.x

        return dx, dy


# 小明去超市买了2个100元一个的苹果，消费税1.1，计算金额
# apple = 100
# apple_num = 2
# tax = 1.1

# # layer
# mul_apple_layer = MulLayer()
# mul_tax_layer = MulLayer()

# # forward
# apple_price = mul_apple_layer.forward(apple, apple_num)
# price = mul_tax_layer.forward(apple_price, tax)

# print('price', price)  # 220

# # backward
# dprice = 1
# dapple_price, dtax = mul_tax_layer.backward(dprice)
# dapple, dapple_num = mul_apple_layer.backward(dapple_price)

# print(dapple, dapple_num, dtax)  # 2.2 110.00000000000001 200


# 加法层
class AddLayer:
    def __init__(self) -> None:
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy


# 购买两个苹果100, 和三个橘子150, 消费税1.1;
# y = (100 * 2 + 150 * 3) * 1.1


apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1


mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
add_tax_layer = MulLayer()


# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = add_tax_layer.forward(all_price, tax)

print(price)  # 715

# backward

dprice = 1
dall_price, dtax = add_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)

print(dapple, dapple_num, dorange, dorange_num, dtax)
# 2.2 110.00000000000001 3.3000000000000003 165.0 650
