import numpy as np
import scipy.signal as signal
import pyaudio
import struct
import scipy.fftpack as spfft
from ipywidgets import ToggleButton, Button
from ipywidgets import HBox
import threading
import cv2

# Signal Processing Parameters
N = 576  # Number of subbands and block size
CHUNK_SIZE = N
FORMAT = pyaudio.paInt16  # Conversion format for PyAudio stream
CHANNELS = 1  # Audio Channels
RATE = 32000  # Sampling Rate in Hz
FFT_LEN = N  # FFT Length

rows = 500
cols = CHUNK_SIZE
fftlen = cols
frame = 0.0 * np.ones((rows, cols, 3))

# The D(z) matrix:
def Dmatrix(samples):
    # implementation of the delay matrix D(z)
    # Delay elements:
    out = np.zeros(N)
    out[0:int(N / 2)] = Dmatrix.z
    Dmatrix.z = samples[0:int(N / 2)]
    out[int(N / 2):N] = samples[int(N / 2):N]
    return out

Dmatrix.z = np.zeros(int(N / 2))

# The inverse D(z) matrix:
def Dmatrixinv(samples):
    # implementation of the delay matrix D(z)
    # Delay elements:
    out = np.zeros(N)
    out[int(N / 2):N] = Dmatrixinv.z
    Dmatrixinv.z = samples[int(N / 2):N]
    out[0:int(N / 2)] = samples[0:int(N / 2)]
    return out

Dmatrixinv.z = np.zeros(int(N / 2))

# The F Matrix:
fcoeff = np.sin(np.pi / (2 * N) * (np.arange(0, 2 * N) + 0.5))
Fmatrix = np.zeros((N, N))
Fmatrix[0:int(N / 2), 0:int(N / 2)] = np.fliplr(np.diag(fcoeff[0:int(N / 2)]))
Fmatrix[int(N / 2):N, 0:int(N / 2)] = np.diag(fcoeff[int(N / 2):N])
Fmatrix[0:int(N / 2), int(N / 2):N] = np.diag(fcoeff[N:int(N + N / 2)])
Fmatrix[int(N / 2):N, int(N / 2):N] = -np.fliplr(np.diag(fcoeff[int(N + N / 2):(2 * N)]))
# The inverse F matrix:
Finv = np.linalg.inv(Fmatrix)

# The DCT4 transform:
def DCT4(samples):
    # use a DCT3 to implement a DCT4:
    samplesup = np.zeros(2 * N)
    # upsample signal:
    samplesup[1::2] = samples
    y = spfft.dct(samplesup, type=3) / 2
    return y[0:N]

# The complete MDCT, Analysis:
def MDCT(samples):
    y = np.dot(samples, Fmatrix)
    y = Dmatrix(y)
    y = DCT4(y)
    return y

# The inverse MDCT, synthesis:
def MDCTinv(y):
    # inverse DCT4 is identical to DCT4:
    x = DCT4(y) * 2 / N
    # inverse D(z) matrix
    x = Dmatrixinv(x)
    # inverse F matrix
    x = np.dot(x, Finv)
    return x

# GUI
toggle_run = ToggleButton(description='Stop')
button_start = Button(description='Start')

def start_button(button_start):
    thread.start()
    button_start.disabled = True

button_start.on_click(start_button)

def on_click_toggle_run(change):
    if change['new'] == False:
        stream.stop_stream()
        stream.close()
        p.terminate()
        cv2.destroyAllWindows()

toggle_run.observe(on_click_toggle_run, 'value')

box_buttons = HBox([button_start, toggle_run])

# Function to Plot MDCT
def run_mdct(toggle_run):
    while True:
        if toggle_run.value == True:
            break

        # Reading from audio input stream into data with block length "CHUNK":
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        # Convert from stream of bytes to a list of short integers (2 bytes here) in "samples":
        shorts = (struct.unpack('h' * CHUNK_SIZE, data))
        samples = np.array(list(shorts), dtype=float)

        # shift "frame" 1 up:
        frame[0:(rows - 1), :] = frame[1:rows, :]
        # compute magnitude of 1D FFT of sound
        # with suitable normalization for the display:
        # frame=np.abs(np.ffqt.fft2(frame[:,:,1]/255.0))/512.0
        # write magnitude spectrum in lowes row of "frame":
        # R=0.25*np.log((np.abs(np.fft.fft(samples[0:fftlen])[0:(fftlen/2)]/np.sqrt(fftlen))+1))/np.log(10.0)

        # This is the FFT of the input:
        # y=np.fft.fft(samples[0:fftlen])
        # This is the analysis MDCT of the input:
        y = MDCT(samples[0:fftlen])

        # yfilt is the processed subbands, processing goes here:
        yfilt = y

        # Waterfall color mapping:
        R = 0.25 * np.log((np.abs(yfilt / np.sqrt(fftlen)) + 1)) / np.log(10.0)
        # Red frame:
        frame[rows - 1, :, 2] = R
        # Green frame:
        frame[rows - 1, :, 1] = np.abs(1 - 2 * R)
        # Blue frame:
        frame[rows - 1, :, 0] = 1.0 - R
        # frame[rows-1,:,0]=frame[rows-1,:,1]**3
        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Inverse FFT:
        # xrek=np.real(np.fft.ifft(yfilt))
        # Inverse/synthesis MDCT:
        xrek = MDCTinv(yfilt).astype(int)
        xrek = np.clip(xrek, -32000, 32000)
        # converting from short integers to a stream of bytes in "data":
        data = struct.pack('h' * len(xrek), *xrek)
        # Writing data back to audio output stream:
        stream.write(data, CHUNK_SIZE)

        # Keep window open until key 'q' is pressed:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    stream.stop_stream()
    stream.close()
    p.terminate()
    cv2.destroyAllWindows()

# Create and start audio stream
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK_SIZE)

# Create a Thread for run_mdct function
thread = threading.Thread(target=run_mdct, args=(toggle_run,))

# Initialize Plot and Display GUI
# display(box_buttons)

if __name__ == '__main__':
    run_mdct(toggle_run)
