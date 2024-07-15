if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy  as np
from dezero import Variable

def rosenbrock(x0,x1):
    y = 100*(x1-x0**2)**2 +(x0-1)**2
    return y   
x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001
iters = 40000
for i in range(iters):
    print(x0,x1)
    y = rosenbrock(x0,x1)
    # 这里cleargrad是因为variable是累加梯度到变量的grad上的
    # 我们是要不停地更新，所以要cleargrad
    x0.cleargrad()
    x1.cleargrad()
    y.backward()
    x0.data -= lr *x0.grad
    x1.data -= lr *x1.grad

# y = rosenbrock(x0,x1)
# y.backward()
# print(x0.grad,x1.grad)