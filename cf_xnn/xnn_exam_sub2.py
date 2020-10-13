import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import s_data_loader_sub as data_loader
dt = data_loader.load_feature_sub2()
# dt = data_loader.load_feature()

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

max_iter = 3200


model = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(), learning_rate='constant',
       learning_rate_init=0.001, max_iter=max_iter, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='lbfgs', tol=0.00001, validation_fraction=0.1,
       verbose=True, warm_start=False)

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


predict_meaure("rfc", model, rx_test)

from s_inspect import inspect_xnn
inspect_xnn(model)