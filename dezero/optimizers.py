import numpy as np
from dezero import cuda
# 将参数更新工作模块化
class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    # 这里的target指的是优化的model
    def setup(self, target):
        self.target = target
        return self

    def update(self):
        # 将None之外的参数汇总到列表
        params = [p for p in self.target.params() if p.grad is not None]
        for f in self.hooks:
            f(params)
        for param in params:
            self.update_one(param)
    # 具体的每一个参数更新是update_one
    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)

class SGD(Optimizer):
    def __init__(self,lr=0.01):
        super().__init__()
        self.lr = lr
    def update_one(self, param):
        param.data -= self.lr *param.grad.data

class MomentumSGD(Optimizer):
    def __init__(self,lr=0.01,momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs={}
    def update_one(self, param):
        v_key = id(param)
        # 对于每一个参数会创建形状上与参数相同的数据
        if v_key not in self.vs:
            xp = cuda.get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)
        v = self.vs[v_key]
        # 更新过程
        v *=self.momentum
        v -=self.lr*param.grad.data
        param.data +=v
           