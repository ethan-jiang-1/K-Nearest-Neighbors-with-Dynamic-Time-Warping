from sklearn.neural_network import MLPClassifier

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import s_data_loader_sub as data_loader
# dt = data_loader.load_feature_time()
dt = data_loader.load_feature_sub3()

# Mapping table for classes
labels = dt.labels
x_train = dt.x_train
y_train = dt.y_train
x_test = dt.x_test
y_test = dt.y_test


skip_ratio = 1
rx_train = x_train[::skip_ratio]
ry_train = y_train[::skip_ratio]
rx_test = x_test[::skip_ratio]
ry_test = y_test[::skip_ratio]

max_iter = 800
tol = 0.0001

model = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(), learning_rate='constant',
       learning_rate_init=0.001, max_iter=max_iter, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='lbfgs', tol=tol, validation_fraction=0.1,
       verbose=False, warm_start=False)


print("training (sub3) iter {} tol {}".format(max_iter, tol))
print("start training...")
model.fit(rx_train, ry_train)
ry_pred=model.predict(rx_test)

from s_confusion import print_confusion_report
print_confusion_report(ry_pred, ry_test, labels)

# from s_confusion import plot_confusion
# plot_confusion(ry_pred, ry_test, labels)

loc_file = __file__
from s_gen_pred import gen_pred
gen_pred(model, rx_test, skip_ratio, loc_file)
print("the training and predict data are all saved in cp_xnn subfolder, run pred to see if java output prediction is ok")
