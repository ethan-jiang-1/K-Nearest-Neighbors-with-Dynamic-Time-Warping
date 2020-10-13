# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix

# from sklearn import svm
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn_porter import Porter

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


skip_ratio = 10
rx_train = x_train[::skip_ratio]
ry_train = y_train[::skip_ratio]
rx_test = x_test[::skip_ratio]
ry_test = y_test[::skip_ratio]


model = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

model.fit(rx_train, ry_train)

if not os.path.isdir("cp_xnn"):
    os.mkdir("cp_xnn")
if not os.path.isdir("cp_xnn/dat"):
    os.mkdir("cp_xnn/dat")


cpm_filename = "cp_xnn/MLPClassifier.java"
print("preper new trained model  {}".format(cpm_filename))
if os.path.isfile(cpm_filename):
    os.unlink(cpm_filename)
porter = Porter(model, language='java')
output = porter.export()
with open(cpm_filename, 'w+') as file:
    n = file.write(output)
    print("traned model saved in c: {} len: {}".format(cpm_filename, len(output)))
if not os.path.isfile(cpm_filename):
    print("Error: no training model saved")
    sys.exit(0)


print("prepare test dat (for predict)...")
for i in range(0, len(rx_test)):
    test_case = rx_test[i]
    cpd_filename = "cp_xnn/dat/{}_{:04d}.tdat".format(len(test_case), i)
    if os.path.isfile(cpd_filename):
        os.unlink(cpd_filename)

    with open(cpd_filename, 'w+') as file:
        for j in range(0, len(test_case)):
            if j == len(test_case) - 1:
                file.write("{:.3f}".format(test_case[j]))
            else:
                file.write("{:.3f} ".format(test_case[j]))
print("total {} test prepared in cp_xnn/dat, each has {} features".format(len(rx_test), len(rx_test[0])))

cpi_filename = "cp_xnn/cp_info.text"
with open(cpi_filename, 'w+') as file:
    file.write("{}\n".format(len(rx_test)))
    file.write("{}\n".format(len(rx_test[0])))
    file.write("{}\n".format(skip_ratio))
    file.write("LinearSVC\n")

cpp_filename = "cp_xnn/pred_result.txt"
print("clean up predict file {}".format(cpp_filename))
if os.path.isfile(cpp_filename):
    os.unlink(cpp_filename)

print("please go to cp_xnn subfolder, and prepare executable: compile {} to class by javac".format(cpm_filename))
