import numpy as np
# import math
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

        self._init_mark_sub1()
        self._init_mark_sub2()        
        self._init_mark_sub3()

    def load(self):
        self._load_from_file()
        self._transform()
        self._normalize()
        dh = DataHolder(self.x_train, self.y_train, self.x_test, self.y_test)
        return dh

    def _load_from_file(self):
        # Import the HAR dataset
        self.x_train_file = open(data_path('train/X_train.txt'), 'r')
        self.y_train_file = open(data_path('train/y_train.txt'), 'r')
        self.x_test_file = open(data_path('test/X_test.txt'), 'r')
        self.y_test_file = open(data_path('test/y_test.txt'), 'r')
     
        # Loop through datasets
        for y in self.y_train_file:
            self.y_train.append(int(y.rstrip('\n')))
        for y in self.y_test_file:
            self.y_test.append(int(y.rstrip('\n')))
        self.y_test = np.array(self.y_test)
        self.y_train = np.array(self.y_train)

    def _transform(self):
        self._transfrim_trim()

        # Convert to numpy for efficiency
        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)

    def _normalize(self):
        pass

    def _transform_merge(self):
        pass

    def _transfrim_trim(self):
        #trunk or not 
        for x in self.x_train_file:
            tmp = [float(ts) for ts in x.split()]
            if self.dt_type.find("_sub") != -1:
                self.x_train.append(self._extract_sub(tmp))
            else:
                self.x_train.append(tmp)
        for x in self.x_test_file:
            #x_test.append([float(ts) for ts in x.split()])
            tmp = [float(ts) for ts in x.split()]
            if self.dt_type.find("_sub") != -1:
                self.x_test.append(self._extract_sub(tmp))
            else:
                self.x_test.append(tmp)

    def _extract_sub(self, tmp):
        xtmp = []
        xmask = None
        if self.dt_type == "feature_sub1":
            xmask = self.mask_sub1
        elif self.dt_type == "feature_sub2":
            xmask = self.mask_sub2        
        elif self.dt_type == "feature_sub3":
            xmask = self.mask_sub3
        if xmask is not None:
            for i in range(0, len(tmp)):
                if i in xmask:
                    xtmp.append(tmp[i])
            return xtmp
        return tmp

    def _init_mark_sub1(self):
        self.mask_sub1 = []
        mask_sub = self.mask_sub1        
        mask_filename = "fsub/sub1_features.txt"
        sf = open(data_path(mask_filename), 'r')
        if sf is not None:
            for fe in sf:
                nm = fe.rstrip('\n').split(' ')
                if len(nm) == 2:
                    try:
                        mask_sub.append(int(nm[0]))
                    except:
                        pass
        
    def _init_mark_sub2(self):
        self.mask_sub2 = []
        mask_sub = self.mask_sub2
        mask_filename = "fsub/sub2_features.txt"        
        sf = open(data_path(mask_filename), 'r')
        if sf is not None:
            for fe in sf:
                nm = fe.rstrip('\n').split(' ')
                if len(nm) == 2:
                    try:
                        mask_sub.append(int(nm[0]))
                    except:
                        pass
        
    def _init_mark_sub3(self):
        self.mask_sub3 = []
        mask_sub = self.mask_sub3
        mask_filename = "fsub/sub3_features.txt"        
        sf = open(data_path(mask_filename), 'r')
        if sf is not None:
            for fe in sf:
                nm = fe.rstrip('\n').split(' ')
                if len(nm) == 2:
                    try:
                        mask_sub.append(int(nm[0]))
                    except:
                        pass

def _load(dt_type):
    print("load data...{}".format(dt_type))
    ds = DataSrc(dt_type)
    dh = ds.load()
    print("load data done {}.".format(dt_type))
    return dh


def load_feature():
    return _load("feature")

def load_feature_sub1():
    return _load("feature_sub1")

def load_feature_sub2():
    return _load("feature_sub2")

def load_feature_sub3():
    return _load("feature_sub3")
