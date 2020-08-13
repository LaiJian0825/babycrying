import librosa
import os
import numpy as np
from tqdm import tqdm
from scipy import signal

DATA_PATH = '../data/train/'
DATA_TEST_PATH = '../data/test'

def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    print(labels)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices


def wav2melgraf(path):
  m_bands = 20
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

  if gram1.shape[0] != 699 or gram.shape[1] != 20 :
    print("eror:!!!!",path)
  return gram1,gram


def save_data_to_array(path=DATA_PATH):
    labels, _ = get_labels(path)

    for label in labels:
        mel_vectors = []
        
        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        #print(wavfiles)
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            
            mel_feat,allwav = wav2melgraf(wavfile)
            mel_vectors.append(mel_feat)
            
        mel_vectors = np.stack(mel_vectors)
        np.save("../npy/MelAndMfcc"+label + '.npy', mel_vectors)
  

def save_data_to_array_test(path=DATA_TEST_PATH):
    mel_vectors = []
        
    wavfiles = [DATA_TEST_PATH + '/' + wavfile for wavfile in os.listdir(DATA_TEST_PATH)]
    for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format('test')):
        mel_feat,allwav = wav2melgraf(wavfile)
        mel_vectors.append(mel_feat)
            
    mel_vectors = np.stack(mel_vectors)
    np.save('../npy/MelAndMfcctest.npy', mel_vectors)
        











save_data_to_array()
save_data_to_array_test()
