from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def print_confusion_report(ry_pred, ry_test, labels):
    print(classification_report(ry_pred, ry_test, target_names=[lb for lb in labels.values()]))


def plot_confusion(ry_pred, ry_test, labels):
    conf_mat = confusion_matrix(ry_pred, ry_test)
    plt.style.use('bmh')
    fig = plt.figure(figsize=(6,6))
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
