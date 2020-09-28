import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import s_data_loader as data_loader
# from s_knn_dtw import KnnDtw

# from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import learning_curve,GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
import keras
from keras.models import Sequential #sequential is required to initialise the neural network
from keras.layers import Dense      #dense is used to build the layers
from keras.layers import Dropout    #Dropout Layer in order to prevent Regularization in the network


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

#creating a network of 561 X 48 X 24 X 12 X6
classifier = Sequential()
classifier.add(Dense(48, input_dim = 561, kernel_initializer='uniform', activation='relu', ))
classifier.add(Dropout(0.1))
classifier.add(Dense(24, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(12, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(6, kernel_initializer='uniform', activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(rx_train, ry_train, batch_size=20, epochs=10, verbose = 4)
label=classifier.predict(rx_test)


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
