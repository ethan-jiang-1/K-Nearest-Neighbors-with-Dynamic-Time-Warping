import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# from sklearn import svm
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn_porter import Porter

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

cpi_filename = "cp_xnn/cp_info.text"
if not os.path.isfile(cpi_filename):
    print("Error no cpi_filename {}".format(cpi_filename))
    sys.exit(-1)

with open(cpi_filename, 'r') as file:
    test_num = int(file.readline().rstrip())
    feature_num = int(file.readline().rstrip())
    skip_ratio = int(file.readline().rstrip())
    model_name = file.readline()

print("CPI", test_num, feature_num, skip_ratio, model_name)

import s_data_loader as data_loader
# dt = data_loader.load_feature_time()
dt = data_loader.load_feature()

# Mapping table for classes
labels = dt.labels
x_train = dt.x_train
y_train = dt.y_train
x_test = dt.x_test
y_test = dt.y_test
rx_train = x_train[::skip_ratio]
ry_train = y_train[::skip_ratio]
rx_test = x_test[::skip_ratio]
ry_test = y_test[::skip_ratio]

cpe_filename = "cp_xnn/MLPClassifier.class"
if not os.path.isfile(cpe_filename):
    print("Error no cpe_filename {}".format(cpe_filename))
    sys.exit(-1)

import subprocess

root_dir = os.getcwd()
working_dir = root_dir + "/cp_xnn"
os.chdir(working_dir)

label_raw = []
for i in range(0, test_num):
    tdat = "dat/{}_{:04d}.tdat".format(feature_num, i)
    with open(tdat, "r") as tdatf:
        line = tdatf.readline()
        nums = line.split(" ")

    # ret, stdout = run_command(["java", "MLPClassifier"], nums, cwd=working_dir)
    ret = True

    cmds = ["java", "MLPClassifier"]
    cmds += nums 
    stdout = subprocess.check_output(cmds)
    print(i, ret, stdout)
    if ret:
        num_result = stdout.decode("utf-8").strip()
        if int(num_result) >= 0:
            label = int(num_result) + 1
            label_raw.append(label)
        else:
            print("error at {} result: {} {}".format(tdat, ret, num_result))
            label_raw.append(1)
    else:
        print("error unknown at {}".foramt(tdat))

os.chdir(root_dir)

label = np.array(label_raw)

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
