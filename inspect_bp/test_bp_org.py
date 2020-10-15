from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def run():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 5000.0  # samples(total points) / T (total time length)
    lowcut = 500.0
    highcut = 1250.0

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

    # Filter a noisy signal.
    T = 0.05  # total time in x-axis
    nsamples = int(T * fs)  # total samples 
    print("nsamples: {} dt {}".format(nsamples, T/nsamples))
    t = np.linspace(0, T, nsamples, endpoint=False)
    # time line (as x-asix: T/nsamples)

    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal ({})'.format(len(x)))

    # filtered output
    order = 3
    y3 = butter_bandpass_filter(x, lowcut, highcut, fs, order=3)
    y6 = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    y9 = butter_bandpass_filter(x, lowcut, highcut, fs, order=9)
    plt.plot(t, y3, label='Filtered signal ({}Hz) oreder ({})'.format(f0, 3))
    plt.plot(t, y6, label='Filtered signal ({}Hz) oreder ({})'.format(f0, 6))
    plt.plot(t, y9, label='Filtered signal ({}Hz) oreder ({})'.format(f0, 9))
    plt.xlabel('time (seconds)')
    # plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()


run()
