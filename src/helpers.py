import random
import numpy as np
from sklearn import metrics

# Define new functions
def protectedDiv(left, right):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


# Define new functions
def Sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x


def test_score(y_test, prediction):
    hamming_loss = metrics.hamming_loss(y_test, prediction)
    f1 = metrics.f1_score(y_test, prediction, average="macro")
    acc = metrics.accuracy_score(y_test, prediction)
    print("Hamming loss", hamming_loss)
    print("F1 Score", f1)
    print("Accuracy ", acc)
    return hamming_loss, f1, acc


def gen_seed() -> int:
    return random.randint(1, 1000)


def set_seed(seed):
    random.seed(43)
    np.random.seed(43)
