# class Function:
#     def __call__(self,input):
#         x = input.data
#         y = self.forward(x)
#         output = Variable(y)
#         return output

#     def forward(self,x):
#         raise NotImplementedError()
    
# class Square(Function):
#     def forward(self, x):
#         return x**2


# 建立了计算图之间的连接
class Variable:
    def __init__(self,data):
        self.data = data
        self.grad = None
        self.creator = None
    def set_creator(self,func):
        self.creator = func
    def backward(self):
        funcs =[self.creator]
        while funcs[0] is not None:
            f = funcs.pop()
            x, y = f.input ,f.output 
            x.grad =f.backward(y.grad)
            print('1')
            if x.creator is not None:
                funcs.append(self.creator)

class Function:
    def __call__(self,input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output
    def forward(self,x):
        raise NotImplementedError

    def backward(self,gy):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):
        # gy 是输出方向传播来的导数
        x = self.input.data
        gx = 2*x*gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = gy*np.exp(x)
        return gx
import numpy as np

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))

a =A(x)
b =B(a)
y =C(b)

y.grad =np.array(1.0)
y.backward()
print(type(y))
print(a.grad)
print(b.grad)
print(x.grad)