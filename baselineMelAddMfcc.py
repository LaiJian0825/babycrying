from common import get_train_test
import os
import numpy as np
from tqdm import tqdm
from keras.utils import to_categorical
from keras.models import load_model
import keras
from sklearn.model_selection import StratifiedKFold
from CnnDataset import mini_XCEPTION,tiny_XCEPTION,ResNet50,CnnOne

DATA_PATH = './data/train/'
DATA_TEST_PATH = './data/test'



X1, Y = get_train_test(DATA_PATH,'/Mel')
X2, Y2 = get_train_test(DATA_PATH,'/')
X = np.concatenate((X1, X2), axis = 1)
print(X1.shape)
print(X.shape)
skf = StratifiedKFold(n_splits=5)

epochs = 100
batch_size = 8
verbose = 1
num_classes = 6
times = 1099
width =20
channel = 1
for idx, (tr_idx, val_idx) in enumerate(skf.split(X, Y)):

    print(idx)
    X_train, X_test = X[tr_idx], X[val_idx]
    y_train, y_test = Y[tr_idx], Y[val_idx]
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], channel) / 255.0
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], channel) / 255.0

    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)
    if idx==1 or idx ==4 :
        model = CnnOne((times, width, channel), num_classes)
    if idx==2 or idx ==0:
        model = tiny_XCEPTION((times, width, channel), num_classes)
    if idx==3:
        model = model = ResNet50((times, width, channel),num_classes)

    model.summary()   
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    my_callbacks = [
        keras.callbacks.EarlyStopping(patience=10),
        keras.callbacks.ModelCheckpoint(filepath='./model/model-{0}.h5'.format(idx), save_best_only=True),
    ]

    model.fit(X_train, y_train_hot, 
              batch_size=batch_size,   
              epochs=epochs, 
              verbose=verbose, 
              validation_data=(X_test, y_test_hot),
              callbacks=my_callbacks
             )




## Test submission
test_pred = np.zeros((228, num_classes))
for path in ['./model/model-0.h5', './model/model-1.h5', './model/model-2.h5','./model/model-3.h5','./model/model-4.h5'][:5]:
    model=load_model(path)
    print(path)
    X_test1 = np.load('./npy/Meltest.npy') / 255.0
    X_test2 = np.load('./npy/test.npy') / 255.0
    X_test = np.concatenate((X_test1, X_test2), axis = 1)
    test_pred += model.predict(X_test.reshape(228, times, width, channel))

wavfiles = [wavfile for wavfile in os.listdir(DATA_TEST_PATH)]   

import pandas as pd
df = pd.DataFrame()
df['id'] = [wavfile for wavfile in os.listdir(DATA_TEST_PATH)]
df['label'] = [['awake','diaper','hug', 'hungry','sleepy', 'uncomfortable'][x] for x in test_pred.argmax(1)]
df.to_csv('tmp225.csv', index=None)