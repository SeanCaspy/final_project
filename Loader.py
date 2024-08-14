import os
import numpy as np
import librosa
import pandas as pd


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)


def load_data(data_dir):
    features = []
    labels = []

    for label in ['Factory', 'NoFactory']:
        folder = os.path.join(data_dir, label)
        for file_name in os.listdir(folder):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder, file_name)
                mfccs = extract_features(file_path)
                features.append(mfccs)
                labels.append(label)

    return np.array(features), np.array(labels)


data_dir = 'dataset'
X, y = load_data(data_dir)

# Save features and labels to a file
np.save('X.npy', X)
np.save('y.npy', y)
