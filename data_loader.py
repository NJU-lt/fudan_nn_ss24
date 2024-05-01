import numpy as np
import os
from sklearn.model_selection import train_test_split
import random
class DataLoader():
    def __init__(self,path):
        self.path = path
    def load_mnist(self,path, kind='train'):
        labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% kind)
        images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% kind)
        with open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)
        with open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 28, 28)  # 关键点
        return images, labels

    def load_data(self):
        train_X, train_Y = self.load_mnist(self.path, kind='train')
        test_X, test_Y = self.load_mnist(self.path, kind='t10k')
        train_X,test_X = train_X.reshape(train_X.shape[0],-1),test_X.reshape(test_X.shape[0],-1)
        train_X = (train_X-train_X.mean())/train_X.std()
        test_X = (test_X-test_X.mean())/test_X.std()
        train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.2,
                                                                              random_state=random.randint(0, 100))
        return train_X, train_Y, val_X, val_Y, test_X, test_Y
