# layer 作为基类出现
from typing import Any
from dezero.core import Parameter
import weakref
import dezero.functions as F
import numpy as np
from dezero import cuda
import os
from dezero.utils import pair
import dezero.functions_conv

# Layer 是 保存参数的类 这些参数继承了Variable的Parameter类


class Layer:
    def __init__(self):
        self._params = set()

    # __setattr__是在设置实例变量时被调用的特殊方法
    # 将layer实例的参数都放进_params这个集合中
    def __setattr__(self, name, value):
        # (Parameter,Layer)使得layer类也能管理layer了
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        # 直接调用父类的方法
        super().__setattr__(name, value)

    # 接受输入并调用forward方法
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x)for x in inputs]
        self.outputs = [weakref.ref(y)for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                # 从layer的layer中递归取出参数
                # yield from 使用一个生成器创建另一个生成器
                yield from obj.params()
            else:
                yield obj
            # 这里的self.__dict__[name]是一个parameter对象
            # yield self.__dict__[name]
            # 由于有yield 所以可以按序取出参数

    def cleargrads(self):
        for param in self.params():
            # 清除的是上面对象的self.W self.b
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()
    # 将parameter 作为一个扁平的、非嵌套的字典取出

    def __flatten_params(self, params_dict, parent_keys=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_keys + '/' + name if parent_keys else name
            if isinstance(obj, Layer):
                obj.__flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weight(self, path):
        # savez只能是numpy,所以要确保数据在内存中
        self.to_cpu()

        params_dict = {}
        self.__flatten_params(params_dict)
        # 创建保存ndarray实例的值的字典
        array_dict = {key: param.data for key,
                      param in params_dict.items() if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self.__flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        # 如果没有指定in_size 那么就会在forward的时候进行权重初始化
        # 而不是在__init__中初始化
        if self.in_size is not None:
            self._init_W()
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1/I)
        self.W.data = W_data

    def forward(self, x):
        # 推迟self.W.data 的初始化，根据in_size来初始化权重参数
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)
        y = F.linear(x, self.W, self.b)
        return y


class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional convolutional layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = dezero.functions_conv.conv2d(
            x, self.W, self.b, self.stride, self.pad)
        return y


class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x)+self.h2h(self.h))
        self.h = h_new
        return h_new

class LSTM(Layer):
    def __init__(self,hidden_size,in_size=None):
        super().__init__()
        H, I = hidden_size, in_size
        self.x2f = Linear(H, in_size=I)
        self.x2i = Linear(H, in_size=I)
        self.x2o = Linear(H, in_size=I)
        self.x2u = Linear(H, in_size=I)
        self.h2f = Linear(H, in_size=H, nobias=True)
        self.h2i = Linear(H, in_size=H, nobias=True)
        self.h2o = Linear(H, in_size=H, nobias=True)
        self.h2u = Linear(H, in_size=H, nobias=True)
        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        if self.h is None:
            f = F.sigmoid(self.x2f(x))
            i = F.sigmoid(self.x2i(x))
            o = F.sigmoid(self.x2o(x))
            u = F.tanh(self.x2u(x))
        else:
            f = F.sigmoid(self.x2f(x) + self.h2f(self.h))
            i = F.sigmoid(self.x2i(x) + self.h2i(self.h))
            o = F.sigmoid(self.x2o(x) + self.h2o(self.h))
            u = F.tanh(self.x2u(x) + self.h2u(self.h))

        if self.c is None:
            c_new = (i * u)
        else:
            c_new = (f * self.c) + (i * u)

        h_new = o * F.tanh(c_new)

        self.h, self.c = h_new, c_new
        return h_new