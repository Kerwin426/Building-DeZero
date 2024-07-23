if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero
import numpy as np
from dezero import test_mode
import dezero.functions as F

x = np.ones(5)
print(x)
y = F.dropout(x)
print(y)
# config中的train=False
with test_mode():
    y = F.dropout(x)
    print(y)
