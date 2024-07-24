if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero.models import VGG16

model = VGG16(pretrained=True)
x = np.random.randn(1,3,224,224).astype(np.float32)
model.plot(x)

