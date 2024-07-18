if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F
import numpy as np