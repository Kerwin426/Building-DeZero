# 获取当前文件的目录并将其父目录添加到模块的搜索路径中
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph

x = Variable(np.array(1.0))
y = (x+3)**2
y.backward()
print(y)
print(x.grad)

def sphere(x,y):
    z = x**2 +y**2 +1
    return z
x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = sphere(x,y)
z.backward()
print(z.data)
print(x.grad,y.grad)


# 做到这里的时候发现之前的函数没有进行x1 = as_array(x1)操作
# 导致一直出现 class int is not supported
def goldstein(x, y):
    z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
        (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return z 
x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein(x,y)
z.backward()
print(x.grad,y.grad)

x.name = 'x'
y.name ='y'
z.name = 'z'

plot_dot_graph(z,verbose=False,to_file='goldstein.png')