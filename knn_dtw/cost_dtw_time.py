
import numpy as np
import matplotlib.pyplot as plt
from s_knn_dtw import KnnDtw
import time as tm

plt.style.use('bmh')

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import s_data_loader as data_loader
# dt = data_loader.load_feature()
# dt = data_loader.load_feature_time()
dt = data_loader.load_feature_freq()

# Mapping table for classes
labels = dt.labels
x_train = dt.x_train
y_train = dt.y_train
x_test = dt.x_test
y_test = dt.y_test

fa = 0
fb = 1
amplitude_a = x_train[fa]
amplitude_b = x_train[fb]
time = np.linspace(0,1,len(amplitude_a))
#amplitude_a = 5*np.sin(time)
#amplitude_b = 3*np.sin(time + 1)

mww = [20, 19, 18, 17, 16, 15, 14, 13, 12,11, 10, 9, 8, 7, 6, 5, 4, 3 ,2, 1]
distances = []
time_taken = []
for i, num in enumerate(mww):
    begin = tm.time()
    distance = KnnDtw(max_warping_window=num)._dtw_distance(amplitude_a, amplitude_b)
    end = tm.time()
    time_taken.append(end - begin)    
    print("cal distance {} {} : {} {}".format(i, num, distance, end-begin))    
    distances.append(distance) 

print("distances {}".format(str(distances)))
print("time {}".format(str(time_taken)))

title = "DTW distance between A and B ({}pt) is {:.2f} ".format(len(amplitude_a), distances[0])
fig = plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
_ = plt.plot(time, amplitude_a, label=labels[y_test[fa]]+str(fa))
_ = plt.plot(time, amplitude_b, label=labels[y_test[fb]]+str(fb))
#_ = plt.plot(mww, distances, label="distace")
#_ = plt.plot(mww, time_taken, label="time")
_ = plt.title(title)
_ = plt.ylabel('Amplitude')
_ = plt.xlabel('Time')
_ = plt.legend()

plt.subplot(3,1,2)
#_ = plt.plot(time, amplitude_a, label='A')
# _ = plt.plot(time, amplitude_b, label='B')
_ = plt.plot(mww, distances, label="distace")
#_ = plt.plot(mww, time_taken, label="time")
#_ = plt.title(title)
_ = plt.ylabel('distance')
_ = plt.xlabel('mww')
_ = plt.legend()

plt.subplot(3,1,3)
#_ = plt.plot(time, amplitude_a, label='A')
# _ = plt.plot(time, amplitude_b, label='B')
#_ = plt.plot(mww, distances, label="distace")
_ = plt.plot(mww, time_taken, label="time")
#_ = plt.title(title)
_ = plt.ylabel('time')
_ = plt.xlabel('mww')
_ = plt.legend()
plt.show()