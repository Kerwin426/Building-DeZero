if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F
import numpy as np

# 对于reshape 需要保证x.data.shape == x.grad.shape

x = Variable(np.array([[1,2,3],[4,5,6]]))
#y = F.reshape(x,(6,))
y = x.reshape([6])
y.backward(create_graph=True)
print(x.grad)

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.transpose(x)
y.backward(create_graph=True)
print(x.grad)