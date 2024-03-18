'''
Only ASK, PAM, PSK, QAM Famliy members needs this processing
'''

import numpy as np

# add carrier for raw digital signal
def add_carrier(symbols, fs = 16e6, fc = 1e6, T = 256e-6, period_per_symbol = 2):
    time = np.arange(0, T-1e-15, 1 / fs)
    carrier_wave = np.exp(1j * 2 * np.pi * fc * time)

    repeat = int(fs/fc*period_per_symbol)
    real, imag = np.real(symbols), np.imag(symbols)
    real, imag = np.repeat(real, repeat), np.repeat(imag, repeat)
    length = len(carrier_wave)
    try:
        modulated_signal = (real[:length]+1j*imag[:length])*carrier_wave
    except Exception as e:
        print(e)
    return modulated_signal


if __name__ == '__main__':
    fs = 16e6
    fc = 1e6
    T = 256e-6
    period_per_symbol = 2 # symbol rate
    print("Point number: {0}, Symbol number: {1}".format(int(T*fs), int(T*fc/period_per_symbol)))

    path = 'D:/torchsig_mine/dataset_digital/qam-16/0000.npy'
    symbols = np.load(path)
    modulated_signal = add_carrier(symbols, fs, fc, T)


