from py_shell import is_in_ipython

print("-------------------------------- section 1")
from IPython.display import Image
Image('images/dtw_knn_schematic.png', width=780)
    

print("-------------------------------- section 2")
Image('images/max_window_warping.png')


print("-------------------------------- section 3")
# import sys
# import collections
# import itertools
import numpy as np
import matplotlib.pyplot as plt

# from scipy.spatial.distance import squareform
plt.style.use('bmh')
# %matplotlib inline

print("-------------------------------- section 4")
from knn_dtw import KnnDtw

time = np.linspace(0,20,1000)
amplitude_a = 5*np.sin(time)
amplitude_b = 3*np.sin(time + 1)

m = KnnDtw()
distance = m._dtw_distance(amplitude_a, amplitude_b)

fig = plt.figure(figsize=(12,4))
_ = plt.plot(time, amplitude_a, label='A')
_ = plt.plot(time, amplitude_b, label='B')
_ = plt.title('DTW distance between A and B is %.2f' % distance)
_ = plt.ylabel('Amplitude')
_ = plt.xlabel('Time')
_ = plt.legend()
if is_in_ipython():
    plt.show()

print("-------------------------------- section 5")
m._dist_matrix(np.random.random((4,50)), np.random.random((4,50)))

print("-------------------------------- section 6")
m._dist_matrix(np.random.random((4,50)), np.random.random((4,50)))

print("-------------------------------- section 7")
# Import the HAR dataset
x_train_file = open('data/UCI-HAR-Dataset/train/X_train.txt', 'r')
y_train_file = open('data/UCI-HAR-Dataset/train/y_train.txt', 'r')

x_test_file = open('data/UCI-HAR-Dataset/test/X_test.txt', 'r')
y_test_file = open('data/UCI-HAR-Dataset/test/y_test.txt', 'r')

# Create empty lists
x_train = []
y_train = []
x_test = []
y_test = []

# Mapping table for classes
labels = {1:'WALKING', 2:'WALKING UPSTAIRS', 3:'WALKING DOWNSTAIRS',
          4:'SITTING', 5:'STANDING', 6:'LAYING'}

# Loop through datasets
for x in x_train_file:
    x_train.append([float(ts) for ts in x.split()])
    
for y in y_train_file:
    y_train.append(int(y.rstrip('\n')))
    
for x in x_test_file:
    x_test.append([float(ts) for ts in x.split()])
    
for y in y_test_file:
    y_test.append(int(y.rstrip('\n')))
    
# Convert to numpy for efficiency
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

print("-------------------------------- section 8")
plt.figure(figsize=(11,7))
colors = ['#D62728','#2C9F2C','#FD7F23','#1F77B4','#9467BD',
          '#8C564A','#7F7F7F','#1FBECF','#E377C2','#BCBD27']

for i, r in enumerate([0,27,65,100,145,172]):
    plt.subplot(3,2,i+1)
    plt.plot(x_train[r][:100], label=labels[y_train[r]], color=colors[i], linewidth=2)
    plt.xlabel('Samples @50Hz')
    plt.legend(loc='upper left')
    plt.tight_layout()
if is_in_ipython():
    plt.show()

print("-------------------------------- section 9")
skip_ratio = 100
m = KnnDtw(n_neighbors=1, max_warping_window=10)
# m.fit(x_train[::10], y_train[::10])
# label, proba = m.predict(x_test[::10])
m.fit(x_train[::skip_ratio], y_train[::skip_ratio])
label, proba = m.predict(x_test[::skip_ratio])

print("-------------------------------- section 10")
from sklearn.metrics import classification_report, confusion_matrix
# print(classification_report(label, y_test[::10], target_names=[l for l in labels.values()]))
print(classification_report(label, y_test[::skip_ratio], target_names=[lb for lb in labels.values()]))

# conf_mat = confusion_matrix(label, y_test[::10])
conf_mat = confusion_matrix(label, y_test[::skip_ratio])

fig = plt.figure(figsize=(6,6))
width = np.shape(conf_mat)[1]
height = np.shape(conf_mat)[0]

res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
for i, row in enumerate(conf_mat):
    for j, c in enumerate(row):
        if c > 0:
            plt.text(j-.2, i+.1, c, fontsize=16)

cb = fig.colorbar(res)
plt.title('Confusion Matrix')
_ = plt.xticks(range(6), [lb for lb in labels.values()], rotation=90)
_ = plt.yticks(range(6), [lb for lb in labels.values()])
if is_in_ipython():
    plt.show()
