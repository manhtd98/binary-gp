import random
import time 
import numpy as np
from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance
from src.helpers import test_score
from src.main import GPClasification
import os 
import json 

def preprocessing(y_train, y_test):
    assert y_train.shape[1]==y_test.shape[1]
    x_tmp = []
    y_tmp = []
    for i in range(y_train.shape[1]):
        if (sum(y_train[:, i])==0) or (sum(y_train[:, i])==y_train.shape[0]) or \
            (sum(y_test[:, i])==0) or (sum(y_test[:, i])==y_test.shape[0]):
            print("Remove label from dataset")
        else:
            x_tmp.append(y_train[:, i])
            y_tmp.append(y_test[:, i])
    x_tmp = np.stack(x_tmp, axis=-1)
    y_tmp = np.stack(y_tmp, axis=-1)
    print(x_tmp.shape)
    return x_tmp, y_tmp 

if __name__ == "__main__":
    datasets = ['Corel5k', 'bibtex' ,'birds', 'delicious' ,'emotions' ,'enron' ,'genbase' ,'mediamill', 'medical', 'rcv1subset1' ,'rcv1subset2' ,'rcv1subset3', 'rcv1subset4', 'rcv1subset5' ,'scene' ,'tmc2007_500' ,'yeast'
              ]
    # datasets = ['emotions']
    import sys
    # dataset = sys.argv[1]
    # out_dir = sys.argv[2]
    # run_idx = int(sys.argv[3])
    # dataset = 'yeast'
    # out_dir = '.'
    # run_idx = 1
    

    # random.seed(1617*run_idx)
    # np.random.seed(1617*run_idx)

    # file_path = out_dir+'%d.txt' % run_idx
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
   
    for dataset in datasets:
        X_train, y_train, feature_names, label_names = load_dataset(dataset, "train")
        X_test, y_test, _, _ = load_dataset(dataset, "test")
        X_train = X_train.toarray()
        y_train = y_train.toarray()

        X_test = X_test.toarray()
        y_test = y_test.toarray()
        num_attr = X_train.shape[1]
        y_train, y_test = preprocessing(y_train, y_test)
        print(dataset)
        k = 0
        for i in range(y_train.shape[1]):
            if (sum(y_train[:, i])==0) or (sum(y_train[:, i])==y_train.shape[0]) or \
            (sum(y_test[:, i])==0) or (sum(y_test[:, i])==y_test.shape[0]):
                k+=1
        print(k)
                # pass

    

