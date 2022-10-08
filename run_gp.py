import itertools
import json
import time
from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance
from src.helpers import test_score
from src.main import GPClasification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from config import model_params
from sklearn.metrics import f1_score
from loguru import logger
from src.helpers import gen_seed
from joblib import Parallel, delayed
import joblib

datasets = (
    "birds",
    "emotions",
    "enron",
    "genbase",
    "medical",
    "yeast",
    "scene",
    "rcv1subset1",
    "tmc2007_500",
)
def load_datasets(dataset_name):
      X_train, y_train, feature_names, label_names = load_dataset(dataset_name, "train")
      X_test, y_test, _, _ = load_dataset("yeast", "test")
      X_train = X_train.toarray()
      y_train = y_train.toarray()  # [:, 0].reshape(-1, 1)

      X_test = X_test.toarray()
      y_test = y_test.toarray()  # [:, 0].reshape(-1, 1)
      # sc = StandardScaler()
      # X_train = sc.fit_transform(X_train)
      # X_test = sc.transform(X_test)
      return X_train, y_train,X_test, y_test


def run_experiments(classifier,dataname, out_file="tmp.json"):
      X_train, y_train,X_test, y_test = load_datasets(dataname)
      start_time = time.time()
      classifier = BinaryRelevance(classifier)
      classifier.fit(X_train, y_train)
      train_time = time.time() - start_time
      prediction = classifier.predict(X_train)
      res_train = test_score(y_train, prediction)

      prediction = classifier.predict(X_test)
      # print(y_test.shape, prediction.shape)
      res_test = test_score(y_test, prediction)
      res = {
            "result_train": res_train,
            "result_test": res_test,
            "training sample": X_train.shape[0],
            "test sample": X_test.shape[0],
            "training time": train_time,
      }
      with open(out_file, "w") as f:
            json.dump(res, f)
      return res


def run_gp(i, dataname):
    logger.info(f"SEED: {i}")
    seed = gen_seed()
    classifier = GPClasification()
    output_name = f"results/{dataname}/gp_{i}_{seed}.json"
    # dtc= DecisionTreeClassifier()
    # NB = GaussianNB()
    # sgd = SGDClassifier(penalty=None)
    # svc = SVC()
    # classifier = KNeighborsClassifier()
    # classifier = LogisticRegression()
    run_experiments(classifier=classifier,dataname= dataname, out_file=output_name)


if __name__ == "__main__":
      number_of_cpu = joblib.cpu_count()
      for data in datasets:
            logger.info(f"RUNNING EXPERIMENCE {data}: {number_of_cpu} CORES")
            delayed_funcs = [delayed(run_gp)(i, data) for i in range(30)]
            parallel_pool = Parallel(n_jobs=number_of_cpu)
            parallel_pool(delayed_funcs)
