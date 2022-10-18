import random
from time import time
import numpy as np
from deap import algorithms, base, creator, tools, gp
from skmultilearn.dataset import load_dataset
from sklearn import metrics
import multiprocessing
from skmultilearn.problem_transform import BinaryRelevance
from src.helpers import test_score
from src.main import GPClasification
import os 
import json 

if __name__ == "__main__":
    # datasets = ['birds',
    #             'emotions',
    #             'enron',
    #             'genbase',
    #             'medical',
    #             'yeast',
    #             'scene',
    #             'rcv1subset1',
    #             'tmc2007_500']
    # datasets = ['emotions']
    import sys
    dataset = sys.argv[1]
    out_dir = sys.argv[2]
    run_idx = int(sys.argv[3])

    random.seed(1617*run_idx)
    np.random.seed(1617*run_idx)

    file_path = out_dir+'%d.txt' % run_idx
    if not os.path.exist(out_dir):
        os.makedirs(out_dir)
    X_train, y_train, feature_names, label_names = load_dataset(dataset, "train")
    X_test, y_test, _, _ = load_dataset(dataset, "test")
    X_train = X_train.toarray()
    y_train = y_train.toarray()

    X_test = X_test.toarray()
    y_test = y_test.toarray()
    num_attr = X_train.shape[1]

    classifier = GPClasification()
    classifier = BinaryRelevance(classifier)
    start_time = time.time()
    classifier.fit(X_train, y_train)
    train_time = time.time() - start_time
    prediction = classifier.predict(X_train)
    res_train = test_score(y_train, prediction)
    res_test = test_score(y_test, prediction)
    res = {
            "result_train": res_train,
            "result_test": res_test,
            "training sample": X_train.shape[0],
            "test sample": X_test.shape[0],
            "training time": train_time,
      }
    with open(file_path, "w") as f:
        json.dump(res, f)


