import numpy as np
import sys
import os 

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir,os.pardir))
sys.path.append(parent_dir)

# #定义变量类
# class Variable:
#     def __init__(self,data):
#         self.data = data

# data = np.array(1.0)
# x = Variable(data)
# print(x.data)


# #定义函数类
# class Function:
#     # 当f=Function()，可以编写f(...)来调用__call__
#     # 是python的一种魔法方法
#     def __call__(self,input):
#         x = input.data
#         y = x**2
#         output = Variable(y)
#         return output
    
# x = Variable(np.array(2.0))
# f = Function()
# y = f(x)
# print(type(y))
# print(y.data)

from dezero import Variable
x = Variable(np.array(1.0))
print(x)