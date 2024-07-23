if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.functions as F
import dezero.functions_conv
from dezero.core import Variable
x1 = np.random.rand(1, 3, 7, 7)
col1 = dezero.functions_conv.im2col(
    x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
print(col1.shape)

# conv2d
N, C, H, W = 1, 5, 15, 15
OC, (KH, KW) = 8, (3, 3)
x = Variable(np.random.randn(N, C, H, W))
W = np.random.randn(OC, C, KH, KW)
y = dezero.functions_conv.conv2d_simple(x, W, b=None, stride=1, pad=1)
y.backward()
print(y.shape)  # (1, 8, 15, 15)
print(x.grad.shape)  # (1, 5, 15, 15)