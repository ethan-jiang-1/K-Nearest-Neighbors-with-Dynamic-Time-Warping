from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def print_confusion_report(ry_pred, ry_test, labels):
    print("\n")
    print("Precision:\t {:.4f}%".format(
        100 * metrics.precision_score(ry_test, ry_pred, average="weighted")))
    print("Recall:\t {:.4f}%".format(
        100 * metrics.recall_score(ry_test, ry_pred, average="weighted")))
    print("f1_score:\t {:.4f}%".format(
        100 * metrics.f1_score(ry_test, ry_pred, average="weighted")))


    print("\n")
    print(classification_report(ry_pred, ry_test, target_names=[lb for lb in labels.values()]))
    print("\n")


def plot_confusion(ry_pred, ry_test, labels, title=None):
    conf_mat = confusion_matrix(ry_pred, ry_test)
    plt.style.use('bmh')
    if title is not None:
        fig_num = title
    else:
        fig_num = "result"

    fig = plt.figure(figsize=(6,6), num=fig_num)
    # width = np.shape(conf_mat)[1]
    # height = np.shape(conf_mat)[0]

    res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            if c > 0:
                plt.text(j-.2, i+.1, c, fontsize=16)

    fig.colorbar(res)
    plt.title('Confusion Matrix')
    _ = plt.xticks(range(6), [lb for lb in labels.values()], rotation=90)
    _ = plt.yticks(range(6), [lb for lb in labels.values()])
    plt.show()
