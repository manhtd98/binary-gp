from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

model_params = {
    'SVM': {
        'model': SVC(gamma='auto'),
        'params' : {
                'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
                'kernel':['linear', 'rbf' ,'poly'],
                'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
        }  
    },
    'Random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10,20,80,100,140,160,180,200]
        }
    },
    'Logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [0.1,0.5,0.8,1,5]
        }
    }
,
    'Naive_bayes_gaussian': {
        'model': GaussianNB(),
        'params': {}
    },
    'SGD_classifier': {
        'model': SGDClassifier(),
        'params': {}
    },
    'KNN_classifier':{
        'model': KNeighborsClassifier(),
        'params' : {}
    },
    'Decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini','entropy'],
            
        }
    }     
}