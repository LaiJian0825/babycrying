# 版本：python 3.6.5，Keras 2.3.1，tensorflow 1.13.1
import librosa
import sklearn
import os
import numpy as np
import re
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten,Conv2D,MaxPooling2D,GlobalAveragePooling2D
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import load_model
import csv
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold, KFold
from keras.optimizers import Adam, RMSprop,SGD
from tqdm import tqdm
from CnnDataset import CnnOne
epochs = 300
batch_size = 32
verbose = 1
num_classes = 6
times = 400
width =128
channel = 1

# MixUP数据增强
class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y




def vgg_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation="relu",padding='same' ,input_shape=(times, width, channel)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=3, activation="relu",padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

    model.add(Conv2D(256, kernel_size=3, activation="relu",padding='same'))
    model.add(Conv2D(256, kernel_size=3, activation="relu",padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

    model.add(Conv2D(512, kernel_size=3, activation="relu", padding='same'))
    model.add(Conv2D(512, kernel_size=3, activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    #model.add(GlobalAveragePooling2D())
    #model.add(Dropout(0.2))
    #
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    #model.add(Dense(256, activation='relu'))

    model.add(Dense(num_classes, activation="softmax"))

    return model


DATA_PATH = './data/train/'
DATA_TEST_PATH = './data/test'


from common import get_train_test
#X1, Y = get_train_test(DATA_PATH,'/Mel')
#X2, Y2 = get_train_test(DATA_PATH,'/')
#X = np.concatenate((X1, X2), axis = 1)
X, Y = get_train_test(DATA_PATH,'/')
skf = StratifiedKFold(n_splits=5)

# 5折交叉验证

n_fold = 0
for idx, (tr_idx, val_idx) in enumerate(skf.split(X, Y)):
    print(idx)


    X_train, X_test = X[tr_idx], X[val_idx]
    y_train, y_test = Y[tr_idx], Y[val_idx]
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], channel) / 255.0
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], channel) / 255.0

    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)
    # 建立模型
    model = vgg_model()
    #model = CnnOne((times, width, channel), num_classes)
    #optimizer = Adam()
    # optimizer=SGD(lr=0.0001,momentum=0.9)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    print(model.summary())
    # 早停函数
    early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=50, mode="auto")
    save_best_model=keras.callbacks.ModelCheckpoint(filepath='./model/model-{0}.h5'.format(idx),monitor='val_accuracy',verbose=0,save_best_only=True,
                                                    save_weights_only=False,mode='auto',period=1)
    ReduceLROnPlateau_func=keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.1,
            patience=30, min_lr=0.0000001, verbose=1)

    # datagen = ImageDataGenerator(
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     horizontal_flip=True)

    # training_generator = MixupGenerator(X_train, y_train_hot, batch_size=batch_size, alpha=0.3, datagen=datagen)()
    # model.fit_generator(generator=training_generator,
    #                     steps_per_epoch=X_train.shape[0] // batch_size,
    #                     validation_data=(X_test, y_test_hot),
    #                     epochs=epochs, verbose=1,
    #                     callbacks=[early_stop, save_best_model, ReduceLROnPlateau_func])
    model.fit(X_train, y_train_hot,
                        batch_size=batch_size, 
                        validation_data=(X_test, y_test_hot),
                        epochs=epochs, verbose=1,
                        callbacks=[early_stop, save_best_model, ReduceLROnPlateau_func])
    # 评估
    _, accuracy = model.evaluate(X_test, y_test_hot)
    print("当前模型正确率为:{}.".format(accuracy))

    
## Test submission
test_pred = np.zeros((228, num_classes))
for path in ['./model/model-0.h5', './model/model-1.h5', './model/model-2.h5','./model/model-3.h5','./model/model-4.h5'][:5]:
    model=load_model(path)
    print(path)
    X_test = np.load('./npy/test.npy') / 255.0
    test_pred += model.predict(X_test.reshape(228, times, width, channel))

wavfiles = [wavfile for wavfile in os.listdir(DATA_TEST_PATH)]   

import pandas as pd
df = pd.DataFrame()
df['id'] = [wavfile for wavfile in os.listdir(DATA_TEST_PATH)]
df['label'] = [['awake','diaper','hug', 'hungry','sleepy', 'uncomfortable'][x] for x in test_pred.argmax(1)]
df.to_csv('tmp225.csv', index=None)