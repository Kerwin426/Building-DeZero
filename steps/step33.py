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
x.cleargrad()
gx.backward()
print(x.grad)
