import random
import time 
import numpy as np
from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance
from src.helpers import test_score
from src.main import GPClasification
import os 
import json 
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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

classifiers = [
    ('knn', KNeighborsClassifier(3)),
    ('svc', SVC(kernel="rbf", C=0.025, probability=True)),
    ('nuvc', NuSVC(probability=True)),
    ('DecisionTreeClassifier', DecisionTreeClassifier()),
    ('RandomForestClassifier', RandomForestClassifier()),
    ('AdaBoostClassifier', AdaBoostClassifier()),
    ('GradientBoostingClassifier', GradientBoostingClassifier()),
    ('GaussianNB', GaussianNB()),
    ]

if __name__ == "__main__":
    
    import sys
    dataset = sys.argv[1]
    out_dir = sys.argv[2]
    run_idx = int(sys.argv[3])
    # dataset = 'yeast'
    # out_dir = '.'
    # run_idx = 6

    
    random.seed(1617)
    np.random.seed(1617)
    name, classifier = classifiers[run_idx]
    file_path = os.path.join(out_dir, name , '%d.json' % run_idx)
    if not os.path.exists(os.path.join(out_dir, name)):
        os.makedirs(os.path.join(out_dir, name))
    X_train, y_train, feature_names, label_names = load_dataset(dataset, "train")
    X_test, y_test, _, _ = load_dataset(dataset, "test")
    X_train = X_train.toarray()
    y_train = y_train.toarray()

    X_test = X_test.toarray()
    y_test = y_test.toarray()
    num_attr = X_train.shape[1]
    y_train, y_test = preprocessing(y_train, y_test)
    res = {}
    if name=='knn':
        best_f1 = 0
        for k in range(3, 10):
            classifier = KNeighborsClassifier(k)
            classifier = BinaryRelevance(classifier)
            start_time = time.time()
            classifier.fit(X_train, y_train)
            train_time = time.time() - start_time
            prediction = classifier.predict(X_train)
            res_train = test_score(y_train, prediction)
            prediction = classifier.predict(X_test)
            res_test = test_score(y_test, prediction)
            if res_test['f1']> best_f1:
                best_f1 = res_test['f1']
                res = {
                    "result_train": res_train,
                    "result_test": res_test,
                    "training sample": X_train.shape[0],
                    "test sample": X_test.shape[0],
                    "training time": train_time,
                    "k": k
        }
    else:
        classifier = BinaryRelevance(classifier)
        start_time = time.time()
        classifier.fit(X_train, y_train)
        train_time = time.time() - start_time
        prediction = classifier.predict(X_train)
        res_train = test_score(y_train, prediction)
        prediction = classifier.predict(X_test)
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


