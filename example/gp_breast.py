from src.main import train_pipeline, evaluation_pipeline
from skmultilearn.dataset import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from utils import get_random_state

get_random_state(42)

convert_dict = {
    "event": {"no-recurrence-events": 0, "recurrence-events": 1},
    "premeno": {"premeno": 0, "lt40": 1, "ge40": 2},
    "row": {"no": 0, "yes": 1},
    "nav": {"left": 0, "right": 1, "central": 2, "no": 3},
    "lowup": {"low": 0, "up": 1, "nan": 0},
}

if __name__ == "__main__":
    # X_train, y_train, feature_names, label_names = load_dataset("emotions", "train")
    # X_test, y_test, _, _ = load_dataset("emotions", "test")
    # X_train = X_train.toarray()
    # y_train = y_train.toarray()[:, 0].reshape(-1, 1)

    # X_test = X_test.toarray()
    # y_test = y_test.toarray()[:, 0].reshape(-1, 1)
    # num_attr = X_train.shape[1]

    df = pd.read_csv("../data/breast-cancer.data", header=None)  # .astype(float)
    df = df.replace("?", "no")
    df[df.columns[0]] = df[df.columns[0]].apply(lambda x: convert_dict["event"][x])
    df[df.columns[2]] = df[df.columns[2]].apply(lambda x: convert_dict["premeno"][x])
    df[df.columns[5]] = df[df.columns[5]].apply(lambda x: convert_dict["row"][x])
    df[df.columns[7]] = df[df.columns[7]].apply(lambda x: convert_dict["nav"][x])
    df[df.columns[9]] = df[df.columns[9]].apply(lambda x: convert_dict["row"][x])

    tmp = df[df.columns[8]].apply(lambda x: x.split("_")).apply(pd.Series)
    tmp.columns = ["left", "up"]
    df[tmp.columns] = tmp
    # df['left'] = df['left'].apply(lambda x: convert_dict["nav"][x])
    # df['up'] = df['up'].apply(lambda x: convert_dict["lowup"][str(x)])
    df = pd.get_dummies(df, columns=["left", "up"])
    df = df.drop(columns=[df.columns[8]])

    tmp = df[df.columns[1]].apply(lambda x: map(int, x.split("-"))).apply(pd.Series)
    tmp.columns = ["left1", "up1"]
    df[tmp.columns] = tmp
    df = df.drop(columns=[df.columns[1]])

    tmp = df[df.columns[2]].apply(lambda x: map(int, x.split("-"))).apply(pd.Series)
    tmp.columns = ["left2", "up2"]
    df[tmp.columns] = tmp
    df = df.drop(columns=[df.columns[2]])
    tmp = df[df.columns[2]].apply(lambda x: map(int, x.split("-"))).apply(pd.Series)
    tmp.columns = ["left3", "up3"]
    df[tmp.columns] = tmp
    df = df.drop(columns=[df.columns[2]])

    df = pd.get_dummies(df, columns=[df.columns[3]])
    X_train = df.drop(columns=[df.columns[6]]).values
    y_train = df.values[:, 6].reshape(-1, 1)
    num_attr = X_train.shape[1]

    sample = int(X_train.shape[0] * 0.2)

    print(X_train[0])
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    toolboxes = train_pipeline(X_train, y_train, num_attr, 300, sample)
    print(X_train.shape, y_train.shape)
    evaluation_pipeline(toolboxes, X_train, y_train, "train", num_attr)
    # evaluation_pipeline(toolboxes, X_test, y_test, "test", num_attr)
