from sklearn.linear_model import LogisticRegression
from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance
from src.helpers import test_score


X_train, y_train, feature_names, label_names = load_dataset("emotions", "train")
X_test, y_test, _, _ = load_dataset("emotions", "test")
X_train = X_train.toarray()
y_train = y_train.toarray()

X_test = X_test.toarray()
y_test = y_test.toarray()


classifier = BinaryRelevance(
    classifier=LogisticRegression(), require_dense=[False, True]
)
prediction = classifier.fit(X_train, y_train).predict(X_test)
test_score(y_test, prediction)
