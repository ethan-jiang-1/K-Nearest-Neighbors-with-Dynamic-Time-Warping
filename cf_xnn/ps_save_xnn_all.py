import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# from s_knn_dtw import KnnDtw

# from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import learning_curve,GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# import keras
# from keras.models import Sequential # sequential is required to initialise the neural network
#from keras.layers import Dense      # dense is used to build the layers
# from keras.layers import Dropout    # Dropout Layer in order to prevent Regularization in the network
from sklearn.neural_network import MLPClassifier

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


model = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

model.fit(rx_train, ry_train)
label=model.predict(rx_test)

import pickle
pkl_filename = "ps_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)
print("train model saved: {}".format(pkl_filename))

# from sklearn_porter import Porter

# porter = Porter(model, language='java')
# output = porter.export()
# cpt_filename = "MLPClassifier.java"
# with open(cpt_filename, 'w') as file:
#     n = file.write(output)
#     print("tran model save in data: {} len: {}".format(cpt_filename, len(output)))

# '''
#             for (int i =0; i< rx_test_num; i++) {
#                 // Features:
#                 double[] features = rx_test[i];
#                 int estimation = clf.predict(features);
#                 System.out.println(estimation);            
#              }
# '''

cpt_testfilename = "MLPClassifier.dat"
if os.path.isfile(cpt_testfilename):
    os.unlink(cpt_testfilename)

print("prepare predit data")

with open(cpt_testfilename, 'w+') as file:
    file.write("int rx_test_num = {};\n".format(len(rx_test)))
    file.write("double rx_test[][] = ")
    file.write(" {\n")
    for i in range(0, len(rx_test)):
        test_case = rx_test[i]
        file.write("{")
        for j in range(0, len(test_case)):
            if j == len(test_case) - 1:
                file.write("{}".format(test_case[j]))
            else:
                file.write("{},".format(test_case[j]))
        file.write("},\n")
    file.write("};\n")

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

from s_inspect import inspect_xnn
inspect_xnn(model)