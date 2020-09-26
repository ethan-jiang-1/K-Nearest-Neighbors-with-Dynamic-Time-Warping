import numpy as np
import matplotlib.pyplot as plt
from s_knn_dtw import KnnDtw
plt.style.use('bmh')
import s_data_loader as data_loader


# dt = data_loader.load_feature()
# dt = data_loader.load_feature_time()
# dt = data_loader.load_feature_freq()
# dt = data_loader.load_raw_acc_x()
# dt = data_loader.load_raw_acc_z()
dt = data_loader.load_raw_acc_o()

# Mapping table for classes
labels = dt.labels
x_train = dt.x_train
y_train = dt.y_train
x_test = dt.x_test
y_test = dt.y_test

plt.figure(figsize=(11,7))
colors = ['#D62728','#2C9F2C','#FD7F23','#1F77B4','#9467BD',
          '#8C564A','#7F7F7F','#1FBECF','#E377C2','#BCBD27']

for i, r in enumerate([0,27,65,100,145,172]):
    plt.subplot(3,2,i+1)
    plt.plot(x_train[r], label=labels[y_train[r]]+str(r), color=colors[i], linewidth=2)
    plt.xlabel('Samples @50Hz')
    plt.legend(loc='upper left')
    plt.tight_layout()

from s_py_shell import is_in_ipython
if is_in_ipython():
    plt.show()
