if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F
import numpy as np

x = Variable(np.array(2.0))
y = x**2
y.backward(create_graph=True)
# gx = x.grad 不仅仅是一个变量还是一个计算图
gx = x.grad
x.cleargrad()
z = gx**3 +y
z.backward()
print(x.grad)