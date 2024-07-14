import numpy as np
import weakref
import contextlib


def as_array(x):
    # 检测是否是标量
    if np.isscalar(x):
        return np.array(x)
    return x


# 只在有限范围内进行禁止反向传播
@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


# 将非Variable类转为Variable类，obj只能是ndarray或Variable类
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Config:
    enable_backprop = True


class Variable:
    __array_priority__ = 200  # 在计算时，优先调用Variable类的运算符，用于处理运算左侧是ndarray实例情况

    def __init__(self, data, name=None):
        # 要求输入一个ndarray的数组
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.name = name
        self.generation = 0

    def __len__(self):
        return len(self.data)

    # 重载print函数
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n'+''*9)  # ndarray转为str 对于换行加9个空格
        return 'variable(' + p + ')'

    # def __mul__(self, other):
    #     return mul(self, other)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    # 当多次处理同一个变量，要重置导数
    def cleargrad(self):
        self.grad = None

    # 利用装饰器property使得shape等方法可以作为实例变量被访问，
    # 这里将ndarray的三个实例变量放入到variable类中

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    def backward(self, retain_grad=False):
        # 不用对最后的dy进行手动设grad为1
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        # 修改funcs的添加逻辑，处理复杂计算图的梯度优先问题
        # funcs = [self.creator]
        # 下面while只支持单个输入输出
        # while funcs:
        #     f = funcs.pop()
        #     x, y = f.input, f.output
        #     x.grad = f.backward(y.grad)
        #     if self.creator is not None:
        #         funcs.append(x.creator)
        funcs = []
        seen_set = set()
        # 调用add_func 函数来添加现在变量的creator seen_set是为了防止重复添加，funcs是为了排序来处理复杂计算图

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # 取出输出的梯度
            gxs = f.backward(*gys)  # 反向传播得到输入的梯度
            # 鉴定是否为元组，或者说数据保存为元组是因为会出现return x1, x2这种类型
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            # 使用zip来设置每一对的导数
            for x, gx in zip(f.inputs, gxs):
                # 这里是用输出端传播的导数进行赋值的，如果是两个一样的变量，那么没有相加而是赋值了两次
                # x.grad = gx
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)
            # 上面代码是在反向传播中对输入的操作，这里是在进行方向传播时 每进行完一次传播
            # 就会将输出的梯度置0
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # 将中间变量的梯度内存删除


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y))for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])  # 设置辈分
            for output in outputs:  # 设置计算图（输出的creator）
                output.set_creator(self)

        # 训练过程需要反向传播求出导数，推理过程只进行正向传播，可以把中间过程扔掉
        self.inputs = inputs
        self.outputs = [weakref.ref(output)for output in outputs]
        #
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, gys):
        raise NotImplementedError


class Add(Function):
    def forward(self, x0, x1):
        y = x1+x0
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x1, x0)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0*x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy*x1, gy*x0


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0-x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0/x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy/x1
        gx1 = gy*(-x0/x1**2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c*x**(c-1)*gy
        return gx


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__mul__ = mul
    Variable.__add__ = add
    Variable.__rmul__ = mul
    Variable.__radd__ = add
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__div__ = div
    Variable.__rdiv__ = rdiv
    Variable.__pow__ = pow
