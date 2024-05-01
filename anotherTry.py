import random

from scipy import signal
from numpy import np


def filter_sbs():
    data = 1000
    b = signal.firwin(150, 0.004)
    z = signal.lfilter_zi(b, 1)
    result = np.zeros(2000)  # Use np.zeros with alias
    for i, x in enumerate(data):
        result[i], z = signal.lfilter(b, 1, [x], zi=z)
    return result


if __name__ == '__main__':
    result = filter_sbs()
