import librosa
import os
import numpy as np
from tqdm import tqdm


DATA_PATH = '../data/train/'
DATA_TEST_PATH = '../data/test'

def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    print(labels)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices


def wav2mfcc(file_path):
    wave, sr = librosa.load(file_path, mono=True, sr=16000)
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    
    mfcc = np.transpose(mfcc, (1, 0))
    print(file_path)
    print(mfcc.shape)
    return mfcc


def save_data_to_array(path=DATA_PATH):
    labels, _ = get_labels(path)

    for label in labels:
        mfcc_vectors = []
        
        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        #print(wavfiles)
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = np.zeros((400, 20))
            mfcc_feat = wav2mfcc(wavfile)[:400,: ]
            mfcc[:mfcc_feat.shape[0], :] = mfcc_feat
            mfcc_vectors.append(mfcc) 
        mfcc_vectors = np.stack(mfcc_vectors)
        np.save("../npy/"+label + '.npy', mfcc_vectors)
  

def save_data_to_array_test(path=DATA_TEST_PATH):
    mfcc_vectors = []
        
    wavfiles = [DATA_TEST_PATH + '/' + wavfile for wavfile in os.listdir(DATA_TEST_PATH)]
    for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format('test')):
        mfcc = np.zeros((400, 20))
        mfcc_feat = wav2mfcc(wavfile)[:400,: ]
        mfcc[:mfcc_feat.shape[0], :] = mfcc_feat
        mfcc_vectors.append(mfcc)
            
    mfcc_vectors = np.stack(mfcc_vectors)
    np.save('../npy/test.npy', mfcc_vectors)
        








#save_data_to_array()
save_data_to_array_test()
