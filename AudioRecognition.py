import time

import pyaudio
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('factory_noise_model.pkl')
scaler = joblib.load('scaler.pkl')

SAMPLE_RATE = 16000
CHUNK = 1024


def extract_features(audio_chunk):
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
    if np.all(audio_data == 0):
        return np.zeros(13)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean


def classify_audio(features):
    features = np.expand_dims(features, axis=0)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return prediction[0]


def real_time_classification():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)
    in_factory = False

    print("Listening...")
    try:
        audio_chunk = stream.read(CHUNK)
        features = extract_features(audio_chunk)
        prediction = classify_audio(features)

        if prediction == 'factory':
            print("factory detected!")
            return True

        if not prediction == 'factory':
            print('not in factory')
            return False
        time.sleep(5)

    except KeyboardInterrupt:
        print("Terminating...")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    real_time_classification()
