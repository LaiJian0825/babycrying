import librosa
from Resnet import ResNet50
import os
from scipy import signal
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
from CNNdataset import mini_XCEPTION
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
DATA_PATH = './data/train/'


def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    print(labels)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


def wav2mfcc(file_path):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    mfcc = librosa.feature.mfcc(wave, sr=8000)
    print(mfcc.shape)
    return mfcc
def wav2melgraf(path):
  m_bands = 128
  s_rate = 16000
  win_length = int(0.1 * s_rate)  # Window length 15ms, 25ms, 50ms, 100ms, 200ms
  hop_length = int(0.01 * s_rate)  # Window shift  10ms
  n_fft = win_length
  y, sr = librosa.load(path, sr=s_rate)
  #进行傅里叶变换
  D = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                window=signal.hamming, center=False)) ** 2
    #提取特征
  S = librosa.feature.melspectrogram(S=D, n_mels=m_bands)
  gram = librosa.power_to_db(S, ref=np.max)
  gram = np.transpose(gram, (1, 0))
  print(gram.shape)
  
  gram1 = gram[:699,:128]

  if gram1.shape[0] != 699 or gram.shape[1] != 128 :
    print("eror:!!!!",path)
  return gram1,gram


def save_data_to_array(path=DATA_PATH):
    labels, _, _ = get_labels(path)

    for label in labels:
        mfcc_vectors = []
        
        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        print(wavfiles)
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            #mfcc = np.zeros((20, 400))
            mfcc_feat,allwav = wav2melgraf(wavfile)
            #mfcc[:, :mfcc_feat.shape[1]] = mfcc_feat
            mfcc_vectors.append(mfcc_feat)
            num = int(allwav.shape[0]/699)
            print(num)
            if(num==2):
                mfcc_vectors.append(allwav[699:1398,:128])
            if(num==3):
                mfcc_vectors.append(allwav[699:1398,:128])
                mfcc_vectors.append(allwav[1398:2097,:128])
            if(num==4):
                mfcc_vectors.append(allwav[699:1398,:128])
                mfcc_vectors.append(allwav[1398:2097,:128])
                mfcc_vectors.append(allwav[2097:2796,:128])
            
         
        mfcc_vectors = np.stack(mfcc_vectors)
            
        
        np.save(label + '.npy', mfcc_vectors)

        
DATA_TEST_PATH = './data/test'
def save_data_to_array_test(path=DATA_TEST_PATH):
    mfcc_vectors = []
        
    wavfiles = [DATA_TEST_PATH + '/' + wavfile for wavfile in os.listdir(DATA_TEST_PATH)]
    for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format('test')):
        #mfcc = np.zeros((20, 400))
        mfcc_feat,wavall= wav2melgraf(wavfile)
        
        mfcc_vectors.append(mfcc_feat)
            
    mfcc_vectors = np.stack(mfcc_vectors)
    np.save('test.npy', mfcc_vectors)
        

def get_train_test(split_ratio=0.8, random_state=42):
    labels, indices, _ = get_labels(DATA_PATH)


    X = np.load(labels[0] + '.npy',allow_pickle = True)
    y = np.zeros(X.shape[0])


    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy',allow_pickle = True)
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    return X, y
    # return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)
def get_model():


    return model







#save_data_to_array()
#save_data_to_array_test()

X, Y = get_train_test()
skf = StratifiedKFold(n_splits=5)

test_pred = np.zeros((228, 6))
for idx, (tr_idx, val_idx) in enumerate(skf.split(X, Y)):
    print(idx)


    channel = 1
    epochs = 50
    batch_size = 16
    verbose = 1
    num_classes = 6

    X_train, X_test = X[tr_idx], X[val_idx]
    y_train, y_test = Y[tr_idx], Y[val_idx]
    print("1111",y_train.shape)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], channel) / 255.0
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], channel) / 255.0

    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)
    #model = ResNet50(input_shape=(699, 128, 1), classes=6)
    #model = mini_XCEPTION((699, 128, channel), num_classes)
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(2, 2), strides=(2, 2),activation='relu', input_shape=(699, 128, channel)))
    model.add(Conv2D(32, kernel_size=(5, 5),  strides=(3, 3),activation='relu'))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25)) 
    model.add(Dense(64, activation='relu'))
    model.add(Dense(6, activation='softmax'))   
    model.summary() 
    #model = get_model()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    my_callbacks = [
        keras.callbacks.EarlyStopping(patience=10),
        keras.callbacks.ModelCheckpoint(filepath='model-{0}.h5'.format(idx), save_best_only=True),
    ]

    model.fit(X_train, y_train_hot, 
              batch_size=batch_size, 
              epochs=epochs, 
              verbose=verbose, 
              validation_data=(X_test, y_test_hot),
              callbacks=my_callbacks
             )


test_pred = np.zeros((228, 6))
for path in ['model-0.h5', 'model-1.h5', 'model-2.h5','model-3.h5','model-4.h5'][:5]:
    model.load_weights(path)
    print(path)
    
    X_test = np.load('test.npy') / 255.0
    test_pred += model.predict(X_test.reshape(228, 699, 128, 1))



wavfiles = [wavfile for wavfile in os.listdir(DATA_TEST_PATH)]   

import pandas as pd
df = pd.DataFrame()

df['id'] = [wavfile for wavfile in os.listdir(DATA_TEST_PATH)]
df['label'] = [['awake','diaper','hug', 'hungry','sleepy', 'uncomfortable'][x] for x in test_pred.argmax(1)]
df.to_csv('tmp225.csv', index=None)