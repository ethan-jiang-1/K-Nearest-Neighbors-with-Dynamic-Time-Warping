from py_shell import is_in_ipython
import numpy as np
import matplotlib.pyplot as plt
from knn_dtw import KnnDtw
import time as tm

plt.style.use('bmh')

time = np.linspace(0,20,1000)
amplitude_a = 5*np.sin(time)
amplitude_b = 3*np.sin(time + 1)

mww = [1000, 100, 80, 50, 20, 10, 5 ,2, 1]
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
fig = plt.figure(figsize=(12,4))
plt.subplot(3,1,1)
_ = plt.plot(time, amplitude_a, label='A')
_ = plt.plot(time, amplitude_b, label='B')
#_ = plt.plot(mww, distances, label="distace")
#_ = plt.plot(mww, time_taken, label="time")
_ = plt.title(title)
_ = plt.ylabel('Amplitude')
_ = plt.xlabel('Time')
_ = plt.legend()


if is_in_ipython():
    plt.show()