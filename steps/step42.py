if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F
import numpy as np

np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2*x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def predict(x):
    y = F.matual(x, W)+b
    return y


def mean_squared_error_simple(x0, x1):
    diff = x0-x1
    return F.sum(diff**2)/len(diff)


lr = 0.1
iters = 100
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr*W.grad.data
    b.data -= lr*b.grad.data
    print(W, b, loss)
