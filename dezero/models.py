import dezero.functions as F
import dezero.layers as L

# 这里如果不加入.layers 会导致循环引用
from dezero.layers import Layer
from dezero import utils


# model 比layer多了一个输出计算图的函数
class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

class MLP(Model):
    # fc_output_size 指定全连接层的输出大小
    def __init__(self,fc_output_sizes,activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i ,out_size in enumerate(fc_output_sizes):
            # 由于MLP 所以都是Linear
            layer = L.Linear(out_size)
            # setattr name value
            setattr(self,'l'+str(i),layer)
            # mlp放入layers
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)
