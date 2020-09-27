import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import s_data_loader as data_loader

from s_knn_dtw import KnnDtw


dt = data_loader.load_feature()
# dt = data_loader.load_raw_acc_x()
# dt = data_loader.load_raw_acc_z()

# Mapping table for classes
labels = dt.labels
x_train = dt.x_train
y_train = dt.y_train
x_test = dt.x_test
y_test = dt.y_test


n_neighbors = 1
#n_neighbors = 2
max_warping_window = 10
#max_warping_window = 4
skip_ratio = 100
print("skip_ratio: {} n_neighors: {} max_waraping_window: {}".format(skip_ratio, n_neighbors, max_warping_window))

rx_train = x_train[::skip_ratio]
ry_train = y_train[::skip_ratio]
rx_test = x_test[::skip_ratio]
ry_test = y_test[::skip_ratio]
print("data: fit: {} {} ".format(len(rx_train), len(ry_train)))
print("data: predict: {} {}".format(len(rx_test), len(ry_test)))

m = KnnDtw(n_neighbors=n_neighbors, max_warping_window=max_warping_window)
m.fit(rx_train, ry_train)
label, proba = m.predict(rx_test)

print(classification_report(label, ry_test, target_names=[lb for lb in labels.values()]))

conf_mat = confusion_matrix(label, ry_test)

plt.style.use('bmh')
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
plt.show()
