import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import s_data_loader as data_loader

from sklearn import svm
from sklearn.model_selection import GridSearchCV

# dt = data_loader.load_feature_time()
dt = data_loader.load_feature()
# dt = data_loader.load_raw_acc_x()
# dt = data_loader.load_raw_acc_z()

# Mapping table for classes
labels = dt.labels
x_train = dt.x_train
y_train = dt.y_train
x_test = dt.x_test
y_test = dt.y_test


rx_train = x_train
ry_train = y_train
rx_test = x_test
ry_test = y_test

classifier=svm.SVC()
parameters=[{'kernel': ['rbf'], 'gamma': [0.001, 0.0001], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
model=GridSearchCV(classifier,parameters,n_jobs=-1,cv=4,verbose=4)
print("perform SVM classifier: parameters: {}".format(str(parameters)))

model.fit(rx_train, ry_train)
label=model.predict(rx_test)


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


def predict_meaure(name, model, rx_test):
    import time
    print("check predict performance {}...".format(name))
    begin = time.time()
    model.predict(rx_test)
    delta = time.time() - begin
    print("check predict perform time_elapse: {:.2f} sec on total {} samples, {:.4f} msec per prediction ".format(delta, len(rx_test), delta*1000/len(rx_test)))


predict_meaure("svm", model, rx_test)