if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.datasets
from dezero.models import VGG16
from PIL import Image
import dezero.utils


# model.plot(x)
url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'
img_path = dezero.utils.get_file(url)
img = Image.open(img_path)

x = VGG16.preprocess(img)
# print(type(x),x.shape)
x = x[np.newaxis]  
# 增加用于小批量处理的轴（3，224，224）-->（1，3，224，224）其实就是加入了N轴
model = VGG16(pretrained=True)
with dezero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)
model.plot(x, to_file='vgg.pdf')
labels = dezero.datasets.ImageNet.labels()
print(labels[predict_id])
