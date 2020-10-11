import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from s_data_loader import data_path

plt.style.use('bmh')

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
    lowcut = 10.0
    highcut = 1000.0
    fs = 50.0 * 128  # 128 points /(1/50) # samples(total points) / T (total time length)
    filter_raw = butter_bandpass_filter(acc_raw, lowcut, highcut, fs, order=3)
    return filter_raw


x_acc_raw_file = open(data_path('train/InertialSignals/body_acc_x_train.txt'), 'r')
y_acc_raw_file = open(data_path('train/InertialSignals/body_acc_y_train.txt'), 'r')
z_acc_raw_file = open(data_path('train/InertialSignals/body_acc_z_train.txt'), 'r')

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
y_train_file = open(data_path('train/y_train.txt'), 'r')
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
plt.figure(figsize=(16,9))
for i, r in enumerate([30, 31, 32, 33, 34, 35]):
    ndx = r
    label_pre = labels[y_train[ndx]]
    color = colors[i]

    for j in range(0,3):
        if j %3 == 0:
            acc_raw = x_acc_raw[ndx]
            label = "x_" + label_pre + str(ndx)
        elif j % 3 == 1:
            acc_raw = y_acc_raw[ndx]
            label = "y_" + label_pre + str(ndx)
        elif j %3 == 2:
            acc_raw = z_acc_raw[ndx]
            label = "z_" + label_pre + str(ndx)
        
        data =acc_raw 

        parameters = {'axes.labelsize': 8,
          'axes.titlesize': 8,
          'legend.fontsize': 8}
        plt.rcParams.update(parameters)   
        plt.subplot(6, 3, i*3 + j +1)
        plt.plot(t, data, label=label, color=color, linewidth=2)
        plt.xlabel('{}:{:.3f}/{:.3f}'.format(len(acc_raw), max(acc_raw), min(acc_raw)), fontsize=8)
        plt.legend(loc='upper left')
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()

plt.show()
