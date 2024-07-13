# 获取当前文件的目录并将其父目录添加到模块的搜索路径中
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

x = Variable(np.array(1.0))
y = (x+3)**2
y.backward()
print(y)
print(x.grad)
