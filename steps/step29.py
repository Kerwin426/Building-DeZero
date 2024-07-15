if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy  as np
from dezero import Variable

# 通过牛顿法来进行优化
# 牛顿法利用了二阶导数的信息来优化

def f(x):
    y = x**4 -2 *x**2
    return y 
def gx2(x):
    return 12*x **2 -4
x = Variable(np.array(2.0))
iters = 10
for i in range(iters):
    print(i,x)
    y = f(x)
    x.cleargrad()
    y.backward()
    x.data -= x.grad /gx2(x.data)