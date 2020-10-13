import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# from sklearn import svm
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import s_data_loader as data_loader
# dt = data_loader.load_feature_time()
dt = data_loader.load_feature()

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


import pickle
pkl_filename = "svm_linesvc.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
print("train model loaded: {}".format(pkl_filename))
model = pickle_model

#model = make_pipeline(StandardScaler(), svm.LinearSVC(random_state=0, tol=1e-5))
#model.fit(rx_train, ry_train)
label=model.predict(rx_test)

#import pickle, os
#pkl_filename = "svm_linesvc.pkl"
#with open(pkl_filename, 'wb') as file:
#    pickle.dump(model, file)
#print("train model saved: {} size: {}".format(pkl_filename, os.path.getsize(pkl_filename)))

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
