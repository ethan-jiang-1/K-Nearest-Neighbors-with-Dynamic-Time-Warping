import numpy as np
import math

class data(object):
    def __init__(self):
        pass

def load_feature():
    return _load("feature")

def load_feature_time():
    return _load("feature_time")

def load_feature_freq():
    return _load("feature_freq")

def load_raw_acc_x():
    return _load("raw_acc_x")

def load_raw_acc_z():
    return _load("raw_acc_z")

def load_raw_acc_o():
    return _load("raw_acc_o")

def _load(dt_type):
    print("load data...{}".format(dt_type))
    dt = data()

    # Create empty lists
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # Import the HAR dataset
    if dt_type == "feature" or dt_type == "feature_time" or dt_type == "feature_freq":
        x_train_file = open('data/UCI-HAR-Dataset/train/X_train.txt', 'r')
        y_train_file = open('data/UCI-HAR-Dataset/train/y_train.txt', 'r')
        x_test_file = open('data/UCI-HAR-Dataset/test/X_test.txt', 'r')
        y_test_file = open('data/UCI-HAR-Dataset/test/y_test.txt', 'r')
 
    elif dt_type == "raw_acc_x":
        x_train_file = open('data/UCI-HAR-Dataset/train/InertialSignals/body_acc_x_train.txt', 'r')
        y_train_file = open('data/UCI-HAR-Dataset/train/y_train.txt', 'r')
        x_test_file = open('data/UCI-HAR-Dataset/test/InertialSignals/body_acc_x_test.txt', 'r')
        y_test_file = open('data/UCI-HAR-Dataset/test/y_test.txt', 'r')

    elif dt_type == "raw_acc_z":
        x_train_file = open('data/UCI-HAR-Dataset/train/InertialSignals/body_acc_z_train.txt', 'r')
        y_train_file = open('data/UCI-HAR-Dataset/train/y_train.txt', 'r')
        x_test_file = open('data/UCI-HAR-Dataset/test/InertialSignals/body_acc_z_test.txt', 'r')
        y_test_file = open('data/UCI-HAR-Dataset/test/y_test.txt', 'r')

    elif dt_type == "raw_acc_o":
        x_train_file_0 = open('data/UCI-HAR-Dataset/train/InertialSignals/body_acc_x_train.txt', 'r')
        x_train_file_1 = open('data/UCI-HAR-Dataset/train/InertialSignals/body_acc_y_train.txt', 'r')
        x_train_file_2 = open('data/UCI-HAR-Dataset/train/InertialSignals/body_acc_z_train.txt', 'r')
        y_train_file = open('data/UCI-HAR-Dataset/train/y_train.txt', 'r')
        x_test_file_0 = open('data/UCI-HAR-Dataset/test/InertialSignals/body_acc_x_test.txt', 'r')
        x_test_file_1 = open('data/UCI-HAR-Dataset/test/InertialSignals/body_acc_y_test.txt', 'r')
        x_test_file_2 = open('data/UCI-HAR-Dataset/test/InertialSignals/body_acc_z_test.txt', 'r')
        y_test_file = open('data/UCI-HAR-Dataset/test/y_test.txt', 'r')      

    # Loop through datasets
        

    for y in y_train_file:
        y_train.append(int(y.rstrip('\n')))
    for y in y_test_file:
        y_test.append(int(y.rstrip('\n')))
    y_test = np.array(y_test)
    y_train = np.array(y_train)

    if dt_type == "raw_acc_o":
        tmp0 = []
        tmp1 = []
        tmp2 = []
        for x in x_train_file_0:
            t0 = [float(ts) for ts in x.split()]
            tmp0.append(t0)
        for x in x_train_file_1:
            t1 = [float(ts) for ts in x.split()]
            tmp1.append(t1)        
        for x in x_train_file_2:
            t2 = [float(ts) for ts in x.split()]
            tmp2.append(t2)        
        for i in range(0, len(tmp0)):
            tlo = [] 
            tl = tmp0[i]
            for j in range(0, len(tl)):
                tlo.append(math.sqrt(tmp0[i][j]*tmp0[i][j] + tmp1[i][j]*tmp1[i][j] + tmp2[i][j]*tmp2[i][j]))
            x_train.append(tlo)

        tmp0 = []
        tmp1 = []
        tmp2 = []
        for x in x_test_file_0:
            t0 = [float(ts) for ts in x.split()]
            tmp0.append(t0)
        for x in x_test_file_1:
            t1 = [float(ts) for ts in x.split()]
            tmp1.append(t1)        
        for x in x_test_file_2:
            t2 = [float(ts) for ts in x.split()]
            tmp2.append(t2)        
        for i in range(0, len(tmp0)):
            tlo = [] 
            tl = tmp0[i]
            for j in range(0, len(tl)):
                tlo.append(math.sqrt(tmp0[i][j]*tmp0[i][j] + tmp1[i][j]*tmp1[i][j] + tmp2[i][j]*tmp2[i][j]))
            x_test.append(tlo)  
    else:
        for x in x_train_file:
            tmp = [float(ts) for ts in x.split()]
            if dt_type == "feature_time":
                x_train.append(tmp[0:265])
            elif dt_type == "feature_freq":
                x_train.append(tmp[266:])
            else:
                x_train.append(tmp)
        for x in x_test_file:
            #x_test.append([float(ts) for ts in x.split()])
            tmp = [float(ts) for ts in x.split()]
            if dt_type == "feature_time":
                x_test.append(tmp[0:265])
            elif dt_type == "feature_freq":
                x_test.append(tmp[266:])
            else:
                x_test.append(tmp)
    # Convert to numpy for efficiency
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    print("load data done.")
    if dt_type == "raw_acc_x" or dt_type == "raw_acc_z" or dt_type == "raw_acc_o":
        print("normalize data")
        for i in range(0, len(x_train)):
            x_train[i] = x_train[i] / np.linalg.norm(x_train[i])
        for i in range(0, len(x_test)):
            x_test[i] = x_test[i] / np.linalg.norm(x_test[i])

    # Mapping table for classes
    labels = {1:'WALKING', 2:'WALKING UPSTAIRS', 3:'WALKING DOWNSTAIRS',
            4:'SITTING', 5:'STANDING', 6:'LAYING'}

    dt.x_train = x_train
    dt.y_train = y_train
    dt.x_test = x_test
    dt.y_test = y_test
    dt.labels = labels
    return dt