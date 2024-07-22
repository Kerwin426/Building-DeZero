if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math
import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import optimizers
import dezero.datasets
import dezero.functions as F
from dezero.models import MLP

# Hyperparameters
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = dezero.datasets.Spiral()
model = MLP((hidden_size,3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
# math.ceil 向上舍入最大的整数
max_iter = math.ceil(data_size/batch_size)
for epoch in range(max_epoch):
    # 对一个序列进行随机排序
    index = np.random.permutation(data_size)
    sum_loss = 0
    for i in range(max_iter):
        # 一个batchsize的index
        batch_index = index[i*batch_size:(i+1)*batch_size]
        # 从train_set中取出一个batchsize
        batch = [train_set[i]for i in batch_index]
        # 分别获取得到x和t
        batch_x = np.array([example[0]for example in batch])
        batch_t = np.array([example[1]for example in batch])

        y = model(batch_x)
        loss = F.softmax_cross_entropy_simple(y,batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data)*len(batch_t)
    avg_loss = sum_loss/data_size
    print('epoch %d ,loss %.2f' %(epoch+1,avg_loss))