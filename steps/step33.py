if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable


def f(x):
    y = x ** 4 - 2*x ** 2
    return y


x = Variable(np.array(2.0))
y = f(x)
# 这里的create_graph 是在反向传播的过程中，要不要启动计算图
# 也就是为了考虑是不是要进行二次求导
y.backward(create_graph=True)
print(x.grad)
gx = x.grad
# 如果不加cleargrad 新的反向传播在variable上会保留上次的结果
# retain_grad = False 
x.cleargrad()
gx.backward()
print(x.grad)


import dezero.functions as F

x = Variable(np.array(1.0))
y = F.sin(x)
y.backward(create_graph=True)
for i in range(3):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    print(x.grad)