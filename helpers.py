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
    print("Hamming loss", metrics.hamming_loss(y_test, prediction))
    print("F1 Score", metrics.f1_score(y_test, prediction, average="macro"))
    print("Accuracy ", metrics.accuracy_score(y_test, prediction))
