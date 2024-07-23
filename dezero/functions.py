import numpy as np
from dezero.core import Function, Variable
from dezero.core import as_variable, as_array
from dezero import utils
from dezero import cuda
import dezero



class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy*cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
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
        y = xp.exp(x)
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
        xp = cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
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

# 线性乘积加偏置
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

# 切片操作 是为了原封不动的传递数据的一部分


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


def softmax_cross_entropy_simple(x, t):
    # t是训练数据 也就是正确类别的编号
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax(x)
    p = clip(p, 1e-15, 1.0)  # 防止log(0) 限制最大最小值
    log_p = log(p)  # softmax概率的数组
    # np.arange(N)-->[0,1,2,...,N-1]

    # 提取出对应于训练数据的模型输出[0,t.data[0]][1,t.data[1]...]

    # log_p = [[-0.416553, -1.418106, -2.318287],
    #      [-2.169846, -0.169846, -3.174802],
    #      [-1.205579, -1.497579, -0.741579]]
    # t.data = [0,1,2] 意味着选择[0,0][1,1][2,2]的数据
    # tlog_p = [-0.416553, -0.169846, -0.741579]

    tlog_p = log_p[np.arange(N), t.data]
    y = -1*sum(tlog_p)/N
    return y

# 不可微，用来计算正确率(识别精度)


def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)
    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))


class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy*mask
        return gx


def relu(x):
    return ReLU()(x)


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)

# dropout只有在训练时是消融的，推理的时候要用全部神经元
def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)
    if dezero.Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0-dropout_ratio).astype(x.dtype)
        y = x*mask / scale
        return y
    else:
        return x
