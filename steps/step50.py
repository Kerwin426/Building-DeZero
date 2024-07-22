if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero.functions as F

y = np.array([[0.2,0.8,0],[0.1,0.9,0],[0.8,0.1,0.1]])
t = np.array([1,2,0])
acc = F.accuracy(y,t)
print(acc)