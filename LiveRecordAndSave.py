import pyaudio
import numpy as np
from scipy.signal import butter, lfilter
import wave
from scipy.fft import fft, ifft



CHUNK = 1024  
FORMAT = pyaudio.paInt16
CHANNELS = 1 
RATE = 44100
noise_duration = 1  
alpha = 0.1 
output_filename = "noise_reduced_recording.wav"


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
    reduced_spectrum = data - smoothed_noise

    b, a = butter_bandpass(lowcut=300, highcut=8000, fs=RATE)
    reduced_spectrum = lfilter(b, a, reduced_spectrum)

    # Perform inverse FFT to get noise-reduced audio
    reduced_data = np.real(ifft(reduced_spectrum))
    return reduced_data


if __name__ == '__main__':
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open streams for input (microphone)
    stream_in = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)

    print("Recording noise...")

    # Record noise profile
    noise_data = np.frombuffer(stream_in.read(CHUNK * noise_duration), dtype=np.int16)
    noise_spectrum = fft(noise_data)

    print("Recording...")

    # Open output WAV file
    wav_file = wave.open(output_filename, 'wb')
    wav_file.setnchannels(CHANNELS)
    wav_file.setsampwidth(p.get_sample_size(FORMAT))
    wav_file.setframerate(RATE)

    try:
        while True:
            data = stream_in.read(CHUNK)
            # Convert data to NumPy array
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Perform FFT
            spectrum = fft(audio_data)

            # Apply noise reduction
            reduced_spectrum = noise_reduction(spectrum, noise_spectrum)

            # Perform inverse FFT to get noise-reduced audio
            reduced_data = np.real(ifft(reduced_spectrum))

            # Write the reduced audio data to the WAV file
            wav_file.writeframes(reduced_data.astype(np.int16).tobytes())

    except KeyboardInterrupt:
        # Handle user interrupting the recording with Ctrl+C
        print("Stopping recording...")

    finally:
        # Always close the stream, terminate PyAudio, and close the WAV file
        stream_in.stop_stream()
        stream_in.close()
        p.terminate()
        wav_file.close()

    print("Finished recording and saved to", output_filename)
