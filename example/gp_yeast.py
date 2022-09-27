from src.main import train_pipeline, evaluation_pipeline
from skmultilearn.dataset import load_dataset, load_from_arff
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import get_random_state

get_random_state(42)

if __name__ == "__main__":
    X_train, y_train, feature_names, label_names = load_dataset("yeast", "train")
    X_test, y_test, _, _ = load_dataset("yeast", "test")
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.toarray())
    # X_train = X_train.toarray()
    y_train = y_train.toarray()  # [:, 0].reshape(-1, 1)
    # print(np.sum(y_train, axis=0)/y_train.shape[0])
    # X_test = X_test.toarray()
    X_test = scaler.transform(X_test.toarray())
    y_test = y_test.toarray()  # [:, 0].reshape(-1, 1)
    num_attr = X_train.shape[1]
    sample = int(X_train.shape[0])  # * 0.1)
    toolboxes = train_pipeline(X_train, y_train, num_attr, 300, sample, 40)
    print(X_train.shape, y_train.shape)
    evaluation_pipeline(toolboxes, X_train, y_train, "train", num_attr)
    evaluation_pipeline(toolboxes, X_test, y_test, "test", num_attr)
