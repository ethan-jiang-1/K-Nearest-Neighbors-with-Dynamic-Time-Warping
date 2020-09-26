import numpy as np
import matplotlib.pyplot as plt
from knn_dtw import KnnDtw
plt.style.use('bmh')

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

plt.figure(figsize=(11,7))
for i, r in enumerate([0,27,65,100,145,172]):
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
        
        data =acc_raw    
        plt.subplot(6,3, i*3 + j +1)
        plt.plot(data, label=label, color=color, linewidth=2)
        plt.xlabel('{}:{:.3f}/{:.3f}'.format(len(acc_raw), max(acc_raw), min(acc_raw)))
        plt.legend(loc='upper left')
        plt.xticks(fontsize=10)
        plt.tight_layout()

from py_shell import is_in_ipython
if is_in_ipython():
    plt.show()
