import numpy as np
import math
import os


def data_path(dat_file):
    path = None
    f1 = "data/UCI-HAR-Dataset/"
    if os.path.isdir(f1):
        os.path.isfile(f1 + dat_file)
        path = os.path.abspath(f1 + dat_file)
    if path is None: 
        f2 = "../data/UCI-HAR-Dataset/"
        if os.path.isdir(f2):
            os.path.isfile(f1 + dat_file)
            path = os.path.abspath(f2 + dat_file)
    if path is None:
        print("Failed find data file for " + dat_file)
    else:
        print("data file located {}".format(path))
    return path  


class DataHolder(object):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.labels = {1:'WALKING', 2:'WALKING UPSTAIRS', 3:'WALKING DOWNSTAIRS',
                       4:'SITTING', 5:'STANDING', 6:'LAYING'}


class DataSrc(object):
    def __init__(self, dt_type):
        self.dt_type = dt_type
        # Create empty lists
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

    def load(self):
        self._load_from_file()
        self._transform()
        self._normalize()
        dh = DataHolder(self.x_train, self.y_train, self.x_test, self.y_test)
        return dh

    def _load_from_file(self):
        # Import the HAR dataset
        if self.dt_type == "feature" or self.dt_type == "feature_time" or self.dt_type == "feature_freq":
            self.x_train_file = open(data_path('train/X_train.txt'), 'r')
            self.y_train_file = open(data_path('train/y_train.txt'), 'r')
            self.x_test_file = open(data_path('test/X_test.txt'), 'r')
            self.y_test_file = open(data_path('test/y_test.txt'), 'r')
     
        elif self.dt_type == "raw_acc_x":
            self.x_train_file = open(data_path('train/InertialSignals/body_acc_x_train.txt'), 'r')
            self.y_train_file = open(data_path('train/y_train.txt'), 'r')
            self.x_test_file = open(data_path('test/InertialSignals/body_acc_x_test.txt'), 'r')
            self.y_test_file = open(data_path('test/y_test.txt'), 'r')

        elif self.dt_type == "raw_acc_y":
            self.x_train_file = open(data_path('train/InertialSignals/body_acc_y_train.txt'), 'r')
            self.y_train_file = open(data_path('train/y_train.txt'), 'r')
            self.x_test_file = open(data_path('test/InertialSignals/body_acc_y_test.txt'), 'r')
            self.y_test_file = open(data_path('test/y_test.txt'), 'r')

        elif self.dt_type == "raw_acc_z":
            self.x_train_file = open(data_path('train/InertialSignals/body_acc_z_train.txt'), 'r')
            self.y_train_file = open(data_path('train/y_train.txt'), 'r')
            self.x_test_file = open(data_path('test/InertialSignals/body_acc_z_test.txt'), 'r')
            self.y_test_file = open(data_path('test/y_test.txt'), 'r')

        elif self.dt_type == "raw_acc_o":
            self.x_train_file_0 = open(data_path('train/InertialSignals/body_acc_x_train.txt'), 'r')
            self.x_train_file_1 = open(data_path('train/InertialSignals/body_acc_y_train.txt'), 'r')
            self.x_train_file_2 = open(data_path('train/InertialSignals/body_acc_z_train.txt'), 'r')
            self.y_train_file = open(data_path('train/y_train.txt'), 'r')
            self.x_test_file_0 = open(data_path('test/InertialSignals/body_acc_x_test.txt'), 'r')
            self.x_test_file_1 = open(data_path('test/InertialSignals/body_acc_y_test.txt'), 'r')
            self.x_test_file_2 = open(data_path('test/InertialSignals/body_acc_z_test.txt'), 'r')
            self.y_test_file = open(data_path('test/y_test.txt'), 'r')      

        # Loop through datasets
        for y in self.y_train_file:
            self.y_train.append(int(y.rstrip('\n')))
        for y in self.y_test_file:
            self.y_test.append(int(y.rstrip('\n')))
        self.y_test = np.array(self.y_test)
        self.y_train = np.array(self.y_train)

    def _transform(self):
        if self.dt_type == "raw_acc_o":
            self._transform_merge()
        else:
            self._transfrim_trim()

        # Convert to numpy for efficiency
        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)

    def _normalize(self):
        if self.dt_type == "raw_acc_x" or self.dt_type == "raw_acc_y" or self.dt_type == "raw_acc_z" or self.dt_type == "raw_acc_o":
            print("normalize data")
            for i in range(0, len(self.x_train)):
                self.x_train[i] = self.x_train[i] / np.linalg.norm(self.x_train[i])
            for i in range(0, len(self.x_test)):
                self.x_test[i] = self.x_test[i] / np.linalg.norm(self.x_test[i])

    def _transform_merge(self):
        if self.dt_type != "raw_acc_o":
            return
        
        # merge or not
        tmp0 = []
        tmp1 = []
        tmp2 = []
        for x in self.x_train_file_0:
            t0 = [float(ts) for ts in x.split()]
            tmp0.append(t0)
        for x in self.x_train_file_1:
            t1 = [float(ts) for ts in x.split()]
            tmp1.append(t1)        
        for x in self.x_train_file_2:
            t2 = [float(ts) for ts in x.split()]
            tmp2.append(t2)        
        for i in range(0, len(tmp0)):
            tlo = [] 
            tl = tmp0[i]
            for j in range(0, len(tl)):
                ox = math.sqrt(tmp0[i][j]*tmp0[i][j] + tmp1[i][j]*tmp1[i][j] + tmp2[i][j]*tmp2[i][j])
                #if tmp2[i][j] < 0:
                #    ox = -ox
                tlo.append(ox)
            self.x_train.append(tlo)

        tmp0 = []
        tmp1 = []
        tmp2 = []
        for x in self.x_test_file_0:
            t0 = [float(ts) for ts in x.split()]
            tmp0.append(t0)
        for x in self.x_test_file_1:
            t1 = [float(ts) for ts in x.split()]
            tmp1.append(t1)        
        for x in self.x_test_file_2:
            t2 = [float(ts) for ts in x.split()]
            tmp2.append(t2)        
        for i in range(0, len(tmp0)):
            tlo = [] 
            tl = tmp0[i]
            for j in range(0, len(tl)):
                ox = math.sqrt(tmp0[i][j]*tmp0[i][j] + tmp1[i][j]*tmp1[i][j] + tmp2[i][j]*tmp2[i][j])
                #if tmp2[i][j] < 0:
                #    ox = -ox
                tlo.append(ox)
            self.x_test.append(tlo)

    def _transfrim_trim(self):
        #trunk or not 
        for x in self.x_train_file:
            tmp = [float(ts) for ts in x.split()]
            if self.dt_type == "feature_time":
                self.x_train.append(tmp[0:265])
            elif self.dt_type == "feature_freq":
                self.x_train.append(tmp[266:])
            else:
                self.x_train.append(tmp)
        for x in self.x_test_file:
            #x_test.append([float(ts) for ts in x.split()])
            tmp = [float(ts) for ts in x.split()]
            if self.dt_type == "feature_time":
                self.x_test.append(tmp[0:265])
            elif self.dt_type == "feature_freq":
                self.x_test.append(tmp[266:])
            else:
                self.x_test.append(tmp)


def _load(dt_type):
    print("load data...{}".format(dt_type))
    ds = DataSrc(dt_type)
    dh = ds.load()
    print("load data done {}.".format(dt_type))
    return dh


def load_feature():
    return _load("feature")

def load_feature_time():
    return _load("feature_time")

def load_feature_freq():
    return _load("feature_freq")

def load_raw_acc_x():
    return _load("raw_acc_x")

def load_raw_acc_y():
    return _load("raw_acc_y")

def load_raw_acc_z():
    return _load("raw_acc_z")

def load_raw_acc_o():
    return _load("raw_acc_o")
