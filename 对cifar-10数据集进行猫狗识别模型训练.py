from keras.datasets import cifar10
import tensorflow as tf
import numpy as np
np.random.seed(10)

(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()
x_img_train_normalize = x_img_train.astype('float32') / 255.0#对数据进行归一化
x_img_test_normalize = x_img_test.astype('float32') / 255.0

from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)#变为one_hot编码
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),#卷积层1
                 input_shape=(32, 32,3),
                 activation='relu',
                 padding='same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))#池化层1
model.add(Conv2D(filters=64, kernel_size=(3, 3),#卷积层2
                 activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))#池化层2
model.add(Flatten())#平滑层
model.add(Dropout(rate=0.25))
model.add(Dense(1024, activation='relu'))#隐藏层

model.add(Dropout(rate=0.25))
model.add(Dense(10, activation='softmax'))#输出层
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x_img_train_normalize, y_label_train_OneHot,
                        validation_split=0.2,
                        epochs=10, batch_size=128, verbose=1)
