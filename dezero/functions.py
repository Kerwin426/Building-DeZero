import numpy as np
from dezero.core import Function
from dezero.core import as_variable
from dezero import utils
from dezero import cuda


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy*cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1-y*y)
        return gx


def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.exp()
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy*y
        return gx


def exp(x):
    return Exp()(x)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        # 需要保存输入变量原本的形状
        self.x_shape = x.shape
        # 这里的reshape是ndarray（numpy）的reshape方法
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        # 反向传播的时候输入输出变量大小一致
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        # 这里加入axes计数
        self.axes = axes

    def forward(self, x):
        # 这里由于x是numpy实例，所以会直接调用numpy的transpose方法
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)

# 这个sum是沿着某一个轴算总和


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        # keepdims 是一个标志位
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(
            gy, self.x_shape, self.axis, self.keepdims)

        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

# 广播 就是将一个列表复制到指定的shape

# broadcast_to 和sum_to 函数互相依赖


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

# utils 中的sum_to 函数会求x的元素之和并将结果形状变成shape的形状


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

# 实现了矩阵的乘法


class MatMul(Function):
    def forward(self, x, W):
        # 调用的是ndarray的dot函数
        y = x.dot(W)
        return y

    # 这里的gx gW都是直接推公式，看反向传播x 和W的grad
    def backward(self, gy):
        x, W = self.inputs
        # .T 会自动调用transpose函数
        gx = matual(gy, W.T)
        gW = matual(x.T, gy)
        return gx, gW


def matual(x, W):
    return MatMul()(x, W)


# 对于第三方函数，可以继承于Function类 减少中间变量的内存占用
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0-x1
        y = (diff**2).sum()/len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0-x1
        gx0 = gy*diff*(2./len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matual(gy, W.T)
        gW = matual(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x*0.5)*0.5+0.5
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        # y = self.outputs[0]得到第一个弱引用
        # y = self.outputs[0]()解引用这个弱引用得到实际的对象
        gx = gy * y * (1-y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


def sigmoid_simple(x):
    x = as_variable(x)
    y = 1/(1+exp(-x))
