from sklearn.linear_model import LogisticRegression
from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance
from src.helpers import test_score


X_train, y_train, feature_names, label_names = load_dataset("emotions", "train")
X_test, y_test, _, _ = load_dataset("emotions", "test")
X_train = X_train.toarray()
y_train = y_train.toarray()[:, 0].reshape(-1, 1)

X_test = X_test.toarray()
y_test = y_test.toarray()[:, 0].reshape(-1, 1)


classifier = LogisticRegression()
# classifier = BinaryRelevance(classifier)
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_train)
test_score(y_train, prediction)

prediction = classifier.predict(X_test)
print(y_test.shape, prediction.shape)
test_score(y_test, prediction)
