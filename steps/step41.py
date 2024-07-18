if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F
import numpy as np

x = Variable(np.random.randn(2, 3))
W = Variable(np.random.randn(3, 4))
y = F.matual(x, W)
y.backward()
print(x.grad.shape)
print(W.grad.shape)
