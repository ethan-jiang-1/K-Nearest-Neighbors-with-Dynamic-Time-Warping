def inspect_xnn(model):
    import numpy as np
    print(model)
    print("layers: {}".format(len(model.coefs_)))
    total_co = 0
    for i in range(0, len(model.coefs_)):
        layer = model.coefs_[i]
        print("layer {} hidden size: {} shape: {}".format(i, len(layer), layer.shape))
        total_co += np.size(layer)
    print("total size of coefs_: {}".format(total_co))
