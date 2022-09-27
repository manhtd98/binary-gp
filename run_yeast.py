import itertools
from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance
from src.helpers import test_score
from src.main import GPClasification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from config import model_params

class GridSearch:
    def __init__(self, model, params):
        self.model = model 
        self.params = params 
    def fit(x_train, y_train):
        prams_iter = list(itertools.product(list1, list2))



def run_experiments(classifier):
    classifier = BinaryRelevance(classifier)
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_train)
    test_score(y_train, prediction)

    prediction = classifier.predict(X_test)
    # print(y_test.shape, prediction.shape)
    test_score(y_test, prediction)


if __name__ == "__main__":
    X_train, y_train, feature_names, label_names = load_dataset("yeast", "train")
    X_test, y_test, _, _ = load_dataset("yeast", "test")
    X_train = X_train.toarray()
    y_train = y_train.toarray()  # [:, 0].reshape(-1, 1)

    X_test = X_test.toarray()
    y_test = y_test.toarray()  # [:, 0].reshape(-1, 1)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # classifier = KNeighborsClassifier()
    # classifier = LogisticRegression()
    classifier = GPClasification()
    # dtc= DecisionTreeClassifier()
    # NB = GaussianNB()
    # sgd = SGDClassifier(penalty=None)
    # svc = SVC()
    # scores= []
    # for model_name, mp in model_params.items():
    #     clf =  GridSearchCV(BinaryRelevance(mp['model']), mp['params'], cv=10, return_train_score=False)
    #     clf.fit(X_train,y_train)
    #     scores.append({
    #     'model': model_name,
    #     'best_score': clf.best_score_,
    #     'best_params': clf.best_params_
    # })



    run_experiments(classifier=classifier)
