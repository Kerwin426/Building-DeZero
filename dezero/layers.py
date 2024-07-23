# layer 作为基类出现
from typing import Any
from dezero.core import Parameter
import weakref
import dezero.functions as F
import numpy as np
from dezero import cuda
import os


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
    
    def __flatten_params(self,params_dict,parent_keys=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_keys +'/' + name if parent_keys else name
            if isinstance(obj,Layer):
                obj.__flatten_params(params_dict,key)
            else:
                params_dict[key] = obj
    
    def save_weight(self,path):
        # savez只能是numpy,所以要确保数据在内存中
        self.to_cpu()

        params_dict = {}
        self.__flatten_params(params_dict)
        # 创建保存ndarray实例的值的字典
        array_dict ={key:param.data for key ,param in params_dict.items() if param is not None}
        try:
            np.savez_compressed(path,**array_dict)
        except(Exception,KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self,path):
        npz = np.load(path)
        params_dict={}
        self.__flatten_params(params_dict)
        for key ,param in params_dict.items():
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

    def _init_W(self,xp=np):
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
