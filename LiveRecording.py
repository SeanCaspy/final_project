import sounddevice as sd
import numpy as np
from scipy import signal

# Define filter parameters
Fs = 44100  # Sampling rate (Hz)
cutoff_freq = 0.004  # Normalized cutoff frequency (0-1)

# Design low-pass filter
b = signal.firwin(150, cutoff_freq)


def filter_audio(indata, frames, time, status, z):
    """ Filters incoming audio data in real-time """
    # Reshape audio data (assuming mono channel)
    audio = indata[:, 0]  # Extract first channel if stereo

    # Perform filtering using lfilter
    filtered_audio, z = signal.lfilter(b, 1, audio, zi=z)

    # You can modify or process the filtered audio here (optional)

    return filtered_audio, z  # Return filtered audio and updated z state


# Initial state for the filter (outside the function)
z = signal.lfilter_zi(b, 1)


# Define callback function for audio processing
def callback(indata, frames, time, status):
    """ This function is called for every block of audio data """
    # Filter the audio data
    filtered_data, z = filter_audio(indata, frames, time, status, z)

    # Play the filtered audio (optional)
    # sd.play(filtered_data, Fs)

    # You can send the filtered data for further processing here


with sd.InputStream(samplerate=Fs, channels=1, callback=callback):
    print('Start filtering audio...')
    while True:  # Infinite loop to keep processing audio
        pass  # Placeholder to prevent the loop from exiting
    print('Audio filtering stopped.')  # This won't be reached due to the infinite loop
