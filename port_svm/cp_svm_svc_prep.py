import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

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

cpt_filename = "cp_svm/cp_model_main.c"
if os.path.isfile(cpt_filename):
	os.unlink(cpt_filename)

print("preper new trained model  {}".format(cpt_filename))
porter = Porter(model, language='c')
output = porter.export()


alter_code = """
/* NOTICE comment out or rename previous main and use this main*/
/* rx_test_num and rx_test are defined in cp_test.h */
/* include "cp_test.h" */

#include "cp_test_data.h"
int main(int argc, const char * argv[]) {

    /* Features: */
    FILE *f = fopen("cp_result.txt", "w+");
    float* features;
    int i;
    for (i = 0; i < rx_test_num; i++) {
        features = &(rx_test[i][0]);
        int result = predict(features);
        printf("%d: %d\\n", i, result);
        fprintf(f, "%d\\n", result);
    }
    fclose(f);

    /* Prediction: */
    return 0;

}
"""

with open(cpt_filename, 'w') as file:
    n = file.write(output)
    print("traned model saved in c: {} len: {}".format(cpt_filename, len(output)))

    file.write(alter_code)

if not os.path.isfile(cpt_filename):
	print("Error: no training model saved")
	sys.exit(0)

cpt_testfilename = "cp_svm/cp_test_data.h"
if os.path.isfile(cpt_testfilename):
	os.unlink(cpt_testfilename)

print("prepare predit data")

with open(cpt_testfilename, 'w+') as file:
	file.write("int rx_test_num = {};\n".format(len(rx_test)))
	file.write("float rx_test[{}][{}] = ".format(len(rx_test), len(rx_test[0])))
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


cpt_resultfilename = "cp_svm/cp_result.txt"
if os.path.isfile(cpt_resultfilename):
	os.unlink(cpt_resultfilename)

print("please go to cp_svm subfolder, and alter cp_model_main.c and compile/run it")
