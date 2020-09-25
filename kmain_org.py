from py_shell import is_in_ipython
import numpy as np
import matplotlib.pyplot as plt
from knn_dtw import KnnDtw
from sklearn.metrics import classification_report, confusion_matrix
import data_loader 


# dt = data_loader.load_feature()
# dt = data_loader.load_raw_acc_x()
dt = data_loader.load_raw_acc_z()

# Mapping table for classes
labels = dt.labels
x_train = dt.x_train
y_train = dt.y_train
x_test = dt.x_test
y_test = dt.y_test


skip_ratio = 100
m = KnnDtw(n_neighbors=1, max_warping_window=10)
m.fit(x_train[::skip_ratio], y_train[::skip_ratio])
label, proba = m.predict(x_test[::skip_ratio])

# print(classification_report(label, y_test[::10], target_names=[l for l in labels.values()]))
print(classification_report(label, y_test[::skip_ratio], target_names=[lb for lb in labels.values()]))

# conf_mat = confusion_matrix(label, y_test[::10])
conf_mat = confusion_matrix(label, y_test[::skip_ratio])

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
if is_in_ipython():
    plt.show()
