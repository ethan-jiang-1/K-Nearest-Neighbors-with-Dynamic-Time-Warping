import os
import sys
import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from knn_dtw import KnnDtw
plt.style.use('bmh')


def signal_exit_handler(sig, frame):
    msg = 'Stop Running as SIGINT/SIGTERM received'
    print(msg)
    sys.exit(0)
    

# signal.signal(signal.SIGINT, signal_exit_handler)
# signal.signal(signal.SIGTERM, signal_exit_handler)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def filter_raw(acc_raw):
    lowcut = 5.0 # HZ
    highcut = 2000.0 # HZ
    fs = 50.0 * 128  # 128 points /(1/50) # samples(total points) / T (total time length)
    filter_raw = butter_bandpass_filter(acc_raw, lowcut, highcut, fs, order=3)
    return filter_raw


x_acc_raw_file = open('data/UCI-HAR-Dataset/train/InertialSignals/body_acc_x_train.txt', 'r')
y_acc_raw_file = open('data/UCI-HAR-Dataset/train/InertialSignals/body_acc_y_train.txt', 'r')
z_acc_raw_file = open('data/UCI-HAR-Dataset/train/InertialSignals/body_acc_z_train.txt', 'r')

# Create empty lists
x_acc_raw = []
for x in x_acc_raw_file:
    x_acc_raw.append([float(ts) for ts in x.split()])
y_acc_raw = []
for x in y_acc_raw_file:
    y_acc_raw.append([float(ts) for ts in x.split()])
z_acc_raw = []
for x in z_acc_raw_file:
    z_acc_raw.append([float(ts) for ts in x.split()])

x_acc_raw = np.array(x_acc_raw)
y_acc_raw = np.array(y_acc_raw)
z_acc_raw = np.array(z_acc_raw)

y_train =[]
y_train_file = open('data/UCI-HAR-Dataset/train/y_train.txt', 'r')
for y in y_train_file:
    y_train.append(int(y.rstrip('\n')))
y_train = np.array(y_train)

colors = ['#D62728','#2C9F2C','#FD7F23','#1F77B4','#9467BD',
          '#8C564A','#7F7F7F','#1FBECF','#E377C2','#BCBD27']
labels = {1:'WALKING', 2:'WALKING UPSTAIRS', 3:'WALKING DOWNSTAIRS',
          4:'SITTING', 5:'STANDING', 6:'LAYING'}

fs = 50.0
nsamples = 128
T = nsamples/fs
t = np.linspace(0, T, nsamples, endpoint=False)
plt.figure(figsize=(16,8))
for i, r in enumerate([0,27,65,0,27,65]):
    ndx = r
    label_pre = labels[y_train[ndx]]
    color = colors[i]

    for j in range(0,3):
        if j %3 == 0:
            acc_raw = x_acc_raw[ndx]
            label = "x_" + label_pre
        elif j % 3 == 1:
            acc_raw = y_acc_raw[ndx]
            label = "y_" + label_pre
        elif j %3 == 2:
            acc_raw = z_acc_raw[ndx]
            label = "z_" + label_pre
        if i >= 3:
            label = "f" + label
            acc_raw = filter_raw(acc_raw)
        
        data =acc_raw    
        plt.subplot(6,3, i*3 + j +1)
        plt.plot(t, data, label=label, color=color, linewidth=2)
        plt.xlabel('{}:{:.3f}/{:.3f}'.format(len(acc_raw), max(acc_raw), min(acc_raw)))
        plt.legend(loc='upper left')
        plt.xticks(fontsize=10)
        plt.tight_layout()

from py_shell import is_in_ipython
if is_in_ipython():
    plt.show()
