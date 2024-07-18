import dezero.functions as F
import dezero.layers as L

# 这里如果不加入.layers 会导致循环引用
from dezero.layers import Layer
from dezero import utils


class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)
