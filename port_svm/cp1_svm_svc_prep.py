# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix

from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn_porter import Porter

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import s_data_loader as data_loader
# dt = data_loader.load_feature_time()
dt = data_loader.load_feature_time()
# dt = data_loader.load_raw_acc_x()
# dt = data_loader.load_raw_acc_z()

# Mapping table for classes
labels = dt.labels
x_train = dt.x_train
y_train = dt.y_train
x_test = dt.x_test
y_test = dt.y_test


skip_ratio = 30
rx_train = x_train[::skip_ratio]
ry_train = y_train[::skip_ratio]
#rx_test = x_test[::skip_ratio]
#ry_test = y_test[::skip_ratio]
rx_test = x_test
ry_test = y_test


model = make_pipeline(StandardScaler(), svm.LinearSVC(random_state=0, tol=1e-5))
model.fit(rx_train, ry_train)


if not os.path.isdir("cp_svm"):
    os.mkdir("cp_svm")
if not os.path.isdir("cp_svm/dat"):
    os.mkdir("cp_svm/dat")


cpm_filename = "cp_svm/cp_model_main.c"
print("preper new trained model  {}".format(cpm_filename))
if os.path.isfile(cpm_filename):
    os.unlink(cpm_filename)
porter = Porter(model, language='c')
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
    cpd_filename = "cp_svm/dat/{}_{:04d}.tdat".format(len(test_case), i)
    if os.path.isfile(cpd_filename):
        os.unlink(cpd_filename)

    with open(cpd_filename, 'w+') as file:
        for j in range(0, len(test_case)):
            if j == len(test_case) - 1:
                file.write("{:.3f}".format(test_case[j]))
            else:
                file.write("{:.3f} ".format(test_case[j]))
print("total {} test prepared in cp_svm/dat, each has {} features".format(len(rx_test), len(rx_test[0])))

cpi_filename = "cp_svm/cp_info.text"
with open(cpi_filename, 'w+') as file:
    file.write("{}\n".format(len(rx_test)))
    file.write("{}\n".format(len(rx_test[0])))
    file.write("LinearSVC\n")

cpp_filename = "cp_svm/pred_result.txt"
print("clean up predict file {}".format(cpp_filename))
if os.path.isfile(cpp_filename):
    os.unlink(cpp_filename)

print("please go to cp_svm subfolder, and prepare executable: compile {} to a.out".format(cpm_filename))
