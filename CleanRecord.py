import pyaudio
import numpy as np
from scipy.signal import butter, lfilter

# Define audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Assuming mono microphone input
RATE = 44100

# Define noise reduction parameters (adjust as needed)
noise_duration = 1  # Record noise in seconds before starting speech
alpha = 0.3  # Smoothing factor for spectral subtraction


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def noise_reduction(data, noise_spectrum):
    # Apply smoothing to noise spectrum
    smoothed_noise = alpha * noise_spectrum + (1 - alpha) * np.abs(data)

    # Perform spectral subtraction
    reduced_data = data - smoothed_noise

    # Apply bandpass filter (optional)
    # b, a = butter_bandpass(lowcut=300, highcut=8000, fs=RATE)
    # reduced_data = lfilter(b, a, reduced_data)

    return reduced_data


if __name__ == '__main__':
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open audio stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording noise...")

    # Record noise profile
    noise_data = np.frombuffer(stream.read(CHUNK * noise_duration), dtype=np.int16)
    noise_spectrum = np.fft.fft(noise_data)

    print("Recording...")

    try:
        while True:
            data = stream.read(CHUNK)
            # Convert data to NumPy array
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Perform FFT
            spectrum = np.fft.fft(audio_data)

            # Apply noise reduction
            reduced_spectrum = noise_reduction(spectrum, noise_spectrum)

            # Perform inverse FFT to get noise-reduced audio
            reduced_data = np.real(np.fft.ifft(reduced_spectrum))

            # Optionally, play the reduced audio
            # stream.write(reduced_data.astype(np.int16).tobytes())

    except KeyboardInterrupt:
        # Handle user interrupting the recording with Ctrl+C
        print("Stopping recording...")

    finally:
        # Always close the stream and terminate PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()

    print("Finished recording")
