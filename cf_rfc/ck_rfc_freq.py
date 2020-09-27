import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import s_data_loader as data_loader

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# dt = data_loader.load_feature_time()
dt = data_loader.load_feature_freq()
# dt = data_loader.load_raw_acc_x()
# dt = data_loader.load_raw_acc_z()

# Mapping table for classes
labels = dt.labels
x_train = dt.x_train
y_train = dt.y_train
x_test = dt.x_test
y_test = dt.y_test

skip_ratio = 20
rx_train = x_train[::skip_ratio]
ry_train = y_train[::skip_ratio]
rx_test = x_test[::skip_ratio]
ry_test = y_test[::skip_ratio]

classifier = RandomForestClassifier()
parameters = {'n_estimators': [10, 100, 1000], 'max_depth': [3, 6, 9], 'max_features': ['auto', 'log2']}
model = GridSearchCV(classifier,parameters,n_jobs=-1,cv=4,scoring='accuracy',verbose=4)

model.fit(rx_train, ry_train)
label = model.predict(rx_test)


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
