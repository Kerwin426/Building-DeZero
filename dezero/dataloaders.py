import math
import random
import numpy as np
from dezero import cuda

# shuffle 每轮训练是否对数据集进行重排
# dataloader的作用是从数据集中创建小批量数据


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size/batch_size)
        self.gpu = gpu
        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    # 在__next__中创建小批量数据

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration
        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i*batch_size:(i+1)*batch_size]
        batch = [self.dataset[i] for i in batch_index]
        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0]for example in batch])
        t = xp.array([example[1]for example in batch])

        self.iteration += 1
        return x, t

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True

    def next(self):
        return self.__next__()

# 针对时序序列的dataloader


class SeqDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, gpu=False):
        # shuffle False的原因是数据重排会打乱数据的顺序，而这里是时序序列
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False,
                         gpu=gpu)

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        # jump是偏移量/步长
        jump = self.data_size // self.batch_size
        # batch_index 是用于取出样本数据的索引
        batch_index = [(i * jump + self.iteration) % self.data_size for i in
                       range(self.batch_size)]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t
