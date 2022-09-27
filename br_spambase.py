from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance
from src.helpers import test_score
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("spambase.csv", header=None)
X_train = df.values[:, :57]
y_train = df.values[:, 57].reshape(-1, 1)
num_attr = X_train.shape[1]
# print(X_train.shape, y_train.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.8, random_state=42
)

classifier = RandomForestClassifier()
# classifier = BinaryRelevance(classifier)
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_train)
test_score(y_train, prediction)

prediction = classifier.predict(X_test)
# print(y_test.shape, prediction.shape)
test_score(y_test, prediction)
