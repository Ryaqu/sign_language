import cv2
import numpy as np
import os

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import cnn_sgn

IMG_SIZE = 96
LR = 1e-3  #.001 learing rate

nb_classes=28

MODEL_NAME = 'handsign.model'

def one_targer(labels_dense,nb_classes):
    targets = np.array(Y).reshape(-1)
    print(targets)
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets


train_data = np.load('train_data.npy', allow_pickle=True)

train = train_data[:]
test = train_data[:100]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]
Y1=one_targer(Y,nb_classes)
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]
test_y1=one_targer(test_y,nb_classes)
test_y=test_y1
Y=Y1

model=cnn_sgn.cnn_model()

model.fit({'input': X}, {'targets': Y}, n_epoch=15, validation_set=({'input': test_x}, {'targets': test_y}), 
snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

score = model.evaluate(test_x, test_y)
print('Test accuarcy: %0.4f%%' % (score[0] * 100))
