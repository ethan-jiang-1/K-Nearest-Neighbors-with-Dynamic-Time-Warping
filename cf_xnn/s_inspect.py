def inspect_xnn(model):
    print(model)
    print("layers: {}".format(len(model.coefs_)))
    for i in range(0, len(model.coefs_)):
        print("layer {} hidden size: {}".format(i, len(model.coefs_[i])))    
